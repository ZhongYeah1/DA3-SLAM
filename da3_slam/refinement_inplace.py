"""In-place inter-submap factor update for pass-2 refinement."""

import numpy as np
import gtsam
from gtsam import SL4
from gtsam.symbol_shorthand import X
from termcolor import colored

from da3_slam.refinement import _estimate_scale_between_submaps


def reoptimize_after_refinement_inplace(solver) -> None:
    """In-place inter-submap factor update + global LM (warm-start from pass-1).

    Step 7 (Graph Construction) is not redone. The existing PoseGraph is
    kept -- same nodes, same Values (pass-1 LM converged solution), same
    prior, same inner-submap factors, same loop-closure factors. The
    ONLY thing that changes is the measurement on each non-LC
    inter-submap BetweenFactorSL4: it is recomputed from the
    pass-2-substituted ``submap.proj_mats`` and ``submap.pointclouds``,
    then replaced in place via ``NonlinearFactorGraph.replace(idx, factor)``.
    Step 8 (Global LM) then re-runs on the modified graph, starting from
    pass-1 Values.

    Skipped:
      - First-submap edges (no inter-submap factor exists).
      - is_loop_closure edges (LC factors stay as pass-1; LC submaps not refined).
      - Edges where either endpoint is an LC submap (not refined; mixing
        pass-1 and pass-2 pointclouds for scale would be inconsistent).

    Args:
        solver: Solver instance with ``.map``, ``.graph``, ``.overlap``,
            ``.scale_method``, and ``._edge_log`` attributes. Each non-LC
            inter-submap entry must carry a 7th element with the factor
            index in ``solver.graph.graph`` (NonlinearFactorGraph).
    """
    print(colored(
        "=== Re-optimizing graph after refinement (in-place factor update) ===",
        "cyan",
    ))

    graph_map = solver.map
    overlap = solver.overlap
    scale_method = solver.scale_method
    pose_graph = solver.graph
    intra_submap_noise = pose_graph.intra_submap_noise

    n_replaced = 0
    n_skipped_lc = 0
    for edge_idx, edge_entry in enumerate(solver._edge_log):
        if len(edge_entry) < 7:
            # First-submap edge or LC edge: no factor index recorded.
            continue
        is_loop_closure = edge_entry[4]
        if is_loop_closure:
            continue

        submap_id_curr = edge_entry[0]
        submap_id_prev = edge_entry[2]
        frame_id_prev = edge_entry[3]
        pass1_scale = edge_entry[5]
        factor_idx = edge_entry[6]

        prior_submap = graph_map.get_submap(submap_id_prev)
        current_submap = graph_map.get_submap(submap_id_curr)

        if prior_submap.get_lc_status() or current_submap.get_lc_status():
            n_skipped_lc += 1
            continue

        match_idx_curr = min(overlap - 1, len(current_submap.pointclouds) - 1)

        scale_pass2 = _estimate_scale_between_submaps(
            prior_submap, current_submap,
            frame_id_prev, match_idx_curr,
            scale_method,
        )

        H_scale = np.diag((scale_pass2, scale_pass2, scale_pass2, 1.0))
        H_overlap_new = (
            np.linalg.inv(prior_submap.proj_mats[frame_id_prev])
            @ current_submap.proj_mats[match_idx_curr]
            @ H_scale
        )

        key_prev = X(submap_id_prev + frame_id_prev)
        key_curr = X(submap_id_curr + match_idx_curr)
        new_factor = gtsam.BetweenFactorSL4(
            key_prev, key_curr, SL4(H_overlap_new), intra_submap_noise,
        )
        pose_graph.graph.replace(int(factor_idx), new_factor)

        print(colored(
            f"  edge {edge_idx}: pass1 scale={pass1_scale:.4f} → pass2 scale={scale_pass2:.4f} "
            f"(factor_idx={factor_idx})",
            "green",
        ))
        n_replaced += 1

    print(colored(
        f"  Replaced {n_replaced} inter-submap factors ({n_skipped_lc} skipped: LC endpoint). "
        f"Re-running LM from pass-1 Values...",
        "cyan",
    ))
    pose_graph.optimize()
    print(colored("=== Re-optimization complete ===", "cyan"))
