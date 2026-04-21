import os
import glob
import time
import argparse

import torch
from tqdm.auto import tqdm
import cv2

import da3_slam.slam_utils as utils
from da3_slam.solver import Solver
from da3_slam.da3_wrapper import DA3Wrapper

parser = argparse.ArgumentParser(description="DA3-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--model_name", type=str, default="da3-small", help="Name of the DA3 model to use")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="logs/poses.txt", help="Path to save the log file")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--strategy", type=str, default="baseline", choices=["baseline", "refine", "large-refine"],
                    help="Pipeline strategy: baseline (current), refine (+ pass 2), large-refine (big chunks + pass 2)")
parser.add_argument("--chunk_size", type=int, default=None, help="Override chunk size (total frames per submap including overlap)")
parser.add_argument("--num_passes", type=int, default=None, help="Override number of DA3 passes (1 or 2)")
parser.add_argument("--scale_method", type=str, default=None, choices=["median", "depth-ransac", "depth-weighted"],
                    help="Override scale estimation method")
parser.add_argument("--pass2_chunk_size", type=int, default=None, help="Reserved for future use (pass-2 re-chunking not yet implemented)")
parser.add_argument("--reopt_method", type=str, default="bounded_clip",
                    choices=["bounded_clip", "inplace", "none"],
                    help="Pass-2 reopt strategy: 'bounded_clip'")


def main():
    """
    Main function that wraps the entire pipeline of DA3-SLAM.
    """
    args = parser.parse_args()

    # Resolve strategy into concrete parameters
    from da3_slam.chunk_strategy import STRATEGY_CONFIGS, safe_chunk_size
    from da3_slam.scale_estimator import STRATEGY_SCALE_DEFAULTS

    strategy_config = STRATEGY_CONFIGS.get(args.strategy, STRATEGY_CONFIGS["baseline"])

    # Apply overrides
    chunk_size = args.chunk_size if args.chunk_size is not None else strategy_config.chunk_size
    overlap = strategy_config.overlap
    if args.chunk_size is not None:
        overlap = max(1, int(chunk_size * strategy_config.overlap / strategy_config.chunk_size))

    # Memory guard
    chunk_size = safe_chunk_size(chunk_size, args.model_name, strategy_config.max_vram_gb)

    num_passes = args.num_passes if args.num_passes is not None else (2 if args.strategy != "baseline" else 1)
    scale_method = args.scale_method if args.scale_method is not None else STRATEGY_SCALE_DEFAULTS.get(args.strategy, "median")

    print(f"Strategy: {args.strategy} | chunk_size={chunk_size}, overlap={overlap}, "
          f"passes={num_passes}, scale={scale_method}")

    # Override submap_size and overlapping_window_size from strategy
    args.submap_size = chunk_size - overlap
    args.overlapping_window_size = overlap

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        scale_method=scale_method,
        overlap=overlap,
    )

    print("Initializing and loading DA3 model...")

    model = DA3Wrapper(model_name=args.model_name, device=device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*"))
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower()
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    total_time = time.time() - total_time_start
    print(image_count, "frames processed")
    print("Total time:", total_time)
    if image_count > 0:
        print(f"Total time for DA3 calls: {solver.da3_timer.total_time:.4f}s")
        print("Average DA3 time per frame:", solver.da3_timer.total_time / image_count)
        print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
        print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
        print("Average backend time per frame:", backend_time.total_time / image_count)
        print("Average total time per frame:", total_time / image_count)
        print("Average FPS:", image_count / total_time)

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    #  Pass 2: Refinement with GTSAM-optimized poses + reopt 
    if num_passes >= 2:
        from da3_slam.refinement import run_refinement_pass, reoptimize_after_refinement
        t_refine_start = time.time()
        run_refinement_pass(
            model=model,
            graph_map=solver.map,
            pose_graph=solver.graph,
        )
        if args.reopt_method == "inplace":
            from da3_slam.refinement_inplace import reoptimize_after_refinement_inplace
            reoptimize_after_refinement_inplace(solver)
        elif args.reopt_method == "none":
            pass  # depth/conf swap only, no graph reopt
        else:
            reoptimize_after_refinement(solver)
        print(f"Refinement + reopt took {time.time() - t_refine_start:.2f} seconds")

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        # Log the full point cloud as one file, used for visualization.
        # solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            solver.map.save_framewise_pointclouds(solver.graph, args.log_path.replace(".txt", "_logs"))

        # Export scene for standalone viewer
        from da3_slam.scene_export import export_scene
        _scene_dir = os.path.join(
            os.path.dirname(args.log_path),
            "scene_" + args.strategy,
        )
        export_scene(solver, strategy_name=args.strategy, output_dir=_scene_dir)


if __name__ == "__main__":
    main()
