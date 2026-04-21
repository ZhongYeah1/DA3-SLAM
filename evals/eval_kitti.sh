#!/bin/bash
# KITTI validation-set evaluation driver.
#
# Iterates the 4 methods over the 13 val drives, writes TUM trajectories +
# depth .npy files, then runs evo_ape + eval_kitti_depth.py to produce
# per-run metrics. Resumable -- skips (method, drive) pairs whose TUM output
# already exists.
#
# Usage:
#     ./evals/eval_kitti.sh              # run all 4 methods on all 13 drives
#     ./evals/eval_kitti.sh da3_baseline # run just one method
#     ./evals/eval_kitti.sh '' drive_name  # run all methods on one drive
#     METHODS=da3_baseline DRIVES=2011_09_29_drive_0026_sync ./evals/eval_kitti.sh

set -o pipefail
# We deliberately do NOT set -e: individual per-drive failures (missing GT frames,
# evo_ape errors, empty grep matches) must not abort the whole 52-run batch.
# Every command that could fail is guarded with `|| { ... }` or `|| true`.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

KITTI_ROOT="${KITTI_ROOT:-/nfs/turbo/coe-jungaocv/siyuanb/github/Depth-Anything-3/data/cut3r_data/kitti_backup/kitti}"
MODEL_NAME="${MODEL_NAME:-da3-small}"
MIN_DISPARITY="${MIN_DISPARITY:-15}"   # KITTI driving needs ~15; office_loop uses 50

# All 13 KITTI val drives (by default)
ALL_DRIVES=(
    2011_09_26_drive_0002_sync
    2011_09_26_drive_0005_sync
    2011_09_26_drive_0013_sync
    2011_09_26_drive_0020_sync
    2011_09_26_drive_0023_sync
    2011_09_26_drive_0036_sync
    2011_09_26_drive_0079_sync
    2011_09_26_drive_0095_sync
    2011_09_26_drive_0113_sync
    2011_09_28_drive_0037_sync
    2011_09_29_drive_0026_sync
    2011_09_30_drive_0016_sync
    2011_10_03_drive_0047_sync
)

ALL_METHODS=(vggt_slam da3_baseline da3_refine da3_refine_inplace da3_refine_none da3_large_refine)

# Scope via env vars or positional args
if [ -n "${DRIVES:-}" ]; then
    IFS=',' read -ra DRIVES_ARR <<< "$DRIVES"
elif [ -n "${2:-}" ]; then
    DRIVES_ARR=("$2")
else
    DRIVES_ARR=("${ALL_DRIVES[@]}")
fi

if [ -n "${METHODS:-}" ]; then
    IFS=',' read -ra METHODS_ARR <<< "$METHODS"
elif [ -n "${1:-}" ]; then
    METHODS_ARR=("$1")
else
    METHODS_ARR=("${ALL_METHODS[@]}")
fi

LOG_DIR="$PROJECT_ROOT/evals/logs/kitti"
GT_DIR="$PROJECT_ROOT/evals/logs/kitti_gt"
RESULTS_CSV="$PROJECT_ROOT/evals/logs/kitti_results.csv"
mkdir -p "$LOG_DIR" "$GT_DIR"

# ---- 1. Ensure GT TUM exists for every drive ----
for drive in "${DRIVES_ARR[@]}"; do
    gt_tum="$GT_DIR/${drive}.tum"
    if [ ! -s "$gt_tum" ]; then
        date_prefix="${drive:0:10}"
        drive_dir="$KITTI_ROOT/$date_prefix/$drive"
        echo "[gt] Converting oxts → TUM for $drive"
        python evals/kitti_oxts_to_tum.py "$drive_dir" "$gt_tum"
    fi
done

# ---- 2. CSV header ----
if [ ! -f "$RESULTS_CSV" ]; then
    echo "Method,Drive,ATE_RMSE,ScaleFactor,LC_Count,TUM_Poses,Depth_Frames,Status" > "$RESULTS_CSV"
fi

# ---- 3. SLAM + evo_ape per (method, drive) ----
for method in "${METHODS_ARR[@]}"; do
    for drive in "${DRIVES_ARR[@]}"; do
        echo ""
        echo "============================================================"
        echo "  $method  on  $drive"
        echo "============================================================"

        tum_path="$LOG_DIR/${method}_${drive}.tum"
        out_dir="$PROJECT_ROOT/docs/eval/kitti/$method/$drive"

        # SLAM run (runner is resumable)
        XFORMERS_DISABLED=1 python evals/kitti_runner.py \
            --method "$method" --drive "$drive" \
            --model_name "$MODEL_NAME" \
            --kitti_root "$KITTI_ROOT" \
            --min_disparity "$MIN_DISPARITY" \
            --tum_path "$tum_path" --out_dir "$out_dir" \
            || { echo "  [SLAM FAIL]"; echo "$method,$drive,,,,,,slam_fail" >> "$RESULTS_CSV"; continue; }

        # Ensure the file is non-empty
        if [ ! -s "$tum_path" ]; then
            echo "  [empty TUM -- skipping eval]"
            echo "$method,$drive,,,,,,empty_tum" >> "$RESULTS_CSV"
            continue
        fi

        # evo_ape with align + scale
        gt_tum="$GT_DIR/${drive}.tum"
        ape_out=$(evo_ape tum "$gt_tum" "$tum_path" -as 2>&1 || true)
        rmse=$(echo "$ape_out"  | { grep -E "^\s*rmse" || true; } | head -1 | sed -E 's/.*rmse[^0-9.-]*([0-9.eE+-]+).*/\1/' || true)
        scale=$(echo "$ape_out" | { grep -Ei "scale"  || true; } | head -1 | sed -E 's/.*[^0-9.eE+-]([0-9.eE+-]+).*/\1/' || true)
        rmse="${rmse:-}"; scale="${scale:-}"

        lc_count=$(cat "$out_dir/loop_closure_count.txt" 2>/dev/null | tr -d '[:space:]' || true)
        lc_count="${lc_count:-0}"
        tum_poses=$(wc -l < "$tum_path" | tr -d '[:space:]' || echo 0)
        depth_frames=$(ls "$out_dir" 2>/dev/null | { grep -c "^depth_frame_.*\.npy$" || true; })
        depth_frames="${depth_frames:-0}"

        echo "  ATE RMSE: ${rmse:---}  Scale: ${scale:---}  LC: $lc_count  Poses: $tum_poses  Depth: $depth_frames"
        echo "$method,$drive,$rmse,$scale,$lc_count,$tum_poses,$depth_frames,ok" >> "$RESULTS_CSV"

        # Depth metrics
        depth_json="$LOG_DIR/${method}_${drive}_depth.json"
        python evals/eval_kitti_depth.py \
            --method "$method" --drive "$drive" \
            --pred_dir "$out_dir" --kitti_root "$KITTI_ROOT" \
            --output "$depth_json" \
            || echo "  [depth eval fail -- continuing]"
    done
done

echo ""
echo "============================================================"
echo "Done. CSV at: $RESULTS_CSV"
echo "Generate summary table with: python evals/process_logs_kitti.py"
echo "============================================================"
