# DA3-SLAM

Dense feed-forward SLAM using [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) for depth and pose estimation, with GTSAM factor graph optimization on the SL(4) manifold and SALAD-based loop closure.

Based on [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT_SPARK) -- replaces the VGGT backbone with DA3 via an adapter pattern (DA3 source code is never modified).

Our presentation video: https://www.youtube.com/watch?v=1Zi687jzQ4Y

See also:
- [`Pipeline.md`](Pipeline.md) -- full two-phase pipeline walkthrough (Phase A stages 1-9, Phase B pose-conditioned refinement).
- [`docs/Results.md`](docs/Results.md) -- KITTI / TUM evaluation, ablation tables, and headline numbers.

## Features

- **Multi-view depth & pose** from DA3's DinoV2 backbone + DualDPT decoder
- **SL(4) graph optimization** via GTSAM -- jointly optimizes rotation, translation, and scale
- **Loop closure** via SALAD image retrieval + DA3 feature cosine similarity verification
- **Multi-pass refinement** -- feed GTSAM-optimized poses back to DA3 for improved depth
- **Configurable strategies** -- `baseline`, `refine`, `large-refine` with pluggable scale estimation
- **Real-time visualization** via Viser (web-based 3D viewer)

## Setup

Requires Python 3.11 and a sibling [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) checkout.

```bash
conda create -n da3slam python=3.11 pip -y
conda activate da3slam
chmod +x setup.sh
./setup.sh
```

`setup.sh` installs: PyTorch, DA3 (from `../Depth-Anything-3`), GTSAM with SL(4) support, SALAD, and this package.

**Note:** Must run with `XFORMERS_DISABLED=1` to avoid GPU compatibility issues with SALAD's DINOv2.

## Quick Start

```bash
# Basic run with da3-small
XFORMERS_DISABLED=1 python main.py --image_folder <path_to_images> --max_loops 1 --model_name da3-small

# With 3D visualization (opens browser at port 8080)
XFORMERS_DISABLED=1 python main.py --image_folder <path_to_images> --vis_map --model_name da3-small
```

## Multi-Pass Strategies

DA3-SLAM supports three pipeline strategies via `--strategy`:

| Strategy | Chunk size | Passes | Scale method | Description |
|----------|-----------|--------|-------------|-------------|
| `baseline` | 17 | 1 | median | Standard single-pass pipeline |
| `refine` | 17 | 2 | median | + second DA3 pass with GTSAM-optimized poses |
| `large-refine` | 80 | 2 | depth-ransac | Larger DA3 context + refinement |

```bash
# Refinement pass (re-runs DA3 with optimized poses)
XFORMERS_DISABLED=1 python main.py --image_folder <data> --strategy refine

# Large chunks + refinement
XFORMERS_DISABLED=1 python main.py --image_folder <data> --strategy large-refine --chunk_size 60

# Override individual parameters
XFORMERS_DISABLED=1 python main.py --image_folder <data> --num_passes 2 --scale_method depth-weighted
```

For the full pipeline walkthrough (Phase A stages 1–9 and the Phase B
pose-conditioned refinement that is DA3-SLAM's core novelty), see
[`Pipeline.md`](Pipeline.md).

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `da3-small` | DA3 variant: `da3-small` (0.08B), `da3-large` (0.35B), `da3-giant` (1.15B), `da3nested-giant-large-1.1` (1.40B) |
| `--strategy` | `baseline` | Pipeline strategy: `baseline`, `refine`, `large-refine` |
| `--submap_size` | 16 | New frames per submap (excluding overlap) |
| `--min_disparity` | 50 | Optical flow threshold for keyframe selection |
| `--conf_threshold` | 25.0 | Confidence percentile filter |
| `--lc_thres` | 0.95 | Loop closure similarity threshold |
| `--vis_map` | off | Enable Viser 3D visualization |
| `--chunk_size` | from strategy | Override total frames per DA3 call |
| `--scale_method` | from strategy | `median`, `depth-ransac`, `depth-weighted` |
| `--num_passes` | from strategy | Number of DA3 inference passes (1 or 2) |
| `--reopt_method` | `bounded_clip` | Pass-2 graph reopt: `bounded_clip`, `inplace`, `none` |

## Project Structure

```
DA3-SLAM/
├── main.py                        # Entry point
├── da3_slam/
│   ├── da3_wrapper.py             # DA3 adapter (__call__ + forward_with_poses)
│   ├── solver.py                  # Submap processing, scale dispatch, graph build
│   ├── chunk_strategy.py          # Strategy presets, VRAM guard
│   ├── scale_estimator.py         # Pluggable scale: median, ransac, weighted
│   ├── refinement.py              # Pass 2: GTSAM poses -> DA3 -> rebuild points
│   ├── graph.py                   # GTSAM SL(4) factor graph
│   ├── submap.py                  # Per-chunk data container
│   ├── map.py                     # Submap collection
│   ├── loop_closure.py            # SALAD image retrieval
│   ├── frame_overlap.py           # LK optical flow keyframe selection
│   ├── viewer.py                  # Viser 3D visualization
│   └── slam_utils.py              # Utilities (decompose_camera, sort, etc.)
├── evals/                         # TUM + KITTI benchmark scripts
├── docs/
│   ├── Results.md                 # KITTI / TUM eval tables and takeaways
|   └── Pipeline.md                # Two-phase pipeline walkthrough
└── third_party/salad/             # SALAD (installed by setup.sh)
```

## Results

Headline KITTI numbers, full ablation tables, and per-strategy takeaways
live in [`docs/Results.md`](docs/Results.md). Short version: at the
`da3nested-giant-large-1.1` backbone, DA3-SLAM beats VGGT-SLAM on
**ATE 12/13, AbsRel 12/13, RMSE_d 11/13, δ<1.25 13/13** across the 13
KITTI validation drives.

## Acknowledgments

- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance)
- [VGGT-SLAM / VGGT-SPARK](https://github.com/MIT-SPARK/VGGT_SPARK) (MIT SPARK Lab)
- [SALAD](https://github.com/serizba/salad) (image retrieval)
- [GTSAM](https://github.com/borglab/gtsam) (factor graph optimization)
