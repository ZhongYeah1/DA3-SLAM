# DA3-SLAM Pipeline

DA3-SLAM has two phases. **Phase A** (stages 1-9) is an incremental per-submap SLAM loop shared with VGGT-SLAM -- only the depth/pose backbone is swapped. **Phase B** (stage 10) is DA3-SLAM's novel contribution: a pose-conditioned refinement pass that feeds the GTSAM-optimized trajectory back into DA3 and optionally re-optimizes the factor graph against the refined geometry.

```
=============  PHASE A -- Incremental per-submap SLAM (stages 1-9)  =============
(Shared with VGGT-SLAM; DA3Wrapper is the only backbone-specific module.)

Input Images
     |
     v
[1. Keyframe Selection]     Lucas-Kanade optical flow (FrameTracker)
     |
     v
[2. DA3 Inference]          Pass 1: depth, conf, extrinsics, intrinsics
     |                      ref_view_strategy="first" → frame 0 = identity
     +---> [3. SALAD Embed] Global descriptors for place recognition
     |
     v
[4. Point Cloud]            Camera-space unprojection (NO extrinsic transform)
     |
     v
[5. Loop Closure]           SALAD retrieval + DA3 attention-based verification
     |
     v
[6. Scale Estimation]       Pluggable: median / depth-ransac / depth-weighted
     |
     v
[7. Graph Construction]     SL(4) nodes + between factors
     |
     v
[8. GTSAM Optimization]     Levenberg-Marquardt on SL(4) manifold
     |
     v
[9. Visualization]          Viser 3D point cloud + cameras
     |
     |  (all submaps built; GTSAM-optimized trajectory ready)
     v  (gated by --num_passes >= 2, i.e. strategy ∈ {refine, large-refine})

=======  PHASE B -- Pose-Conditioned Refinement  (DA3-SLAM NOVELTY)  ========
Runs once as a batch after the full Phase-A trajectory is built.
Files: da3_slam/refinement.py, refinement_inplace.py, da3_wrapper.forward_with_poses

[10a. Pose Extraction]      H_submap → K @ inv(H) → decompose → c2w → w2c
                            (extract_poses_from_graph)
     |
     v
[10b. DA3 forward_with_poses]
                            CameraEnc injects 9-dim pose tokens (t, quat, FoV)
                            into DinoV2 at layer alt_start; normalizes extrinsics
                            (first frame → I, t /= median dist); deliberately
                            SKIPS Umeyama realignment.
     |
     v
[10c. Scale Alignment]      scale = mean|t_pass1| / mean|t_pass2|  (skip frame 0)
                            clamp [0.01, 100.0]; apply to depth AND to t_pass2.
     |
     v
[10d. Substitute]           submap.poses / proj_mats / pointclouds / colors /
                            conf / conf_masks all replaced with pass-2 values.
     |
     v
[10e. Graph Reopt]          --reopt_method selects one of:
                              * bounded_clip (default) -- rebuild graph with
                                per-edge scale clipped to pass-1 × [1 ± 0.05]
                              * inplace               -- NonlinearFactorGraph.
                                replace inter-submap factors + warm-start LM
                              * none                  -- skip reopt entirely
                                (empirically best on KITTI driving)
     |
     v
refined submap geometry  +  (optionally) refined trajectory
```

**Why Stage 10 is the focus of DA3-SLAM.** Stages 1–9 are structurally inherited from VGGT-SLAM; none of them are novel. Stage 10 is DA3-SLAM's original contribution -- it closes the SLAM loop from "poses improve with depth" back to "depth improves with poses" using DA3's pose-conditioning API (which VGGT does not expose). Sub-steps 10a–10e, the `forward_with_poses` wrapper, the scale-alignment rule, and the three reopt variants (with their KITTI characterization) are all original to this project.
