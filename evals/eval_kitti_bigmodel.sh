#!/bin/bash
# KITTI big-model sweep driver.
#
# Runs 5 (experiment, method, model) combos across all 13 KITTI val drives:
#   da3_baseline_giant             -- da3_baseline       + da3-giant
#   da3_refine_none_giant          -- da3_refine_none    + da3-giant
#   da3_baseline_nested            -- da3_baseline       + da3nested-giant-large
#   da3_refine_none_nested         -- da3_refine_none    + da3nested-giant-large
#   da3_refine_inplace_nested      -- da3_refine_inplace + da3nested-giant-large
#
# Same CSV + depth_json conventions as eval_kitti_ablation.sh. Resumable via
# kitti_runner.py's existing check (TUM + depth manifest).
#
# Usage:
#   ./evals/eval_kitti_bigmodel.sh                         # run all 65 pairs
#   DRIVES=2011_09_26_drive_0002_sync ./evals/eval_kitti_bigmodel.sh
#   EXPERIMENTS=da3_baseline_giant ./evals/eval_kitti_bigmodel.sh

set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure da3slam env is active. The runner imports depth_anything_3, which
# is only installed in da3slam. Caller may either activate it manually or
# rely on this auto-activation via the conda shell integration.
if [ -z "$CONDA_PREFIX" ] || [ "$(basename "$CONDA_PREFIX")" != "da3slam" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || echo /nfs/turbo/coe-jungaocv/siyuanb/miniconda3)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate da3slam
fi

KITTI_ROOT="${KITTI_ROOT:-/nfs/turbo/coe-jungaocv/siyuanb/github/Depth-Anything-3/data/cut3r_data/kitti_backup/kitti}"
MIN_DISPARITY="${MIN_DISPARITY:-15}"

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

# 3-field colon-separated tuples: experiment_name:method_key:model_name
ALL_EXPERIMENTS=(
    "da3_baseline_giant:da3_baseline:da3-giant"
    "da3_refine_none_giant:da3_refine_none:da3-giant"
    "da3_baseline_nested:da3_baseline:da3nested-giant-large"
    "da3_refine_none_nested:da3_refine_none:da3nested-giant-large"
    "da3_refine_inplace_nested:da3_refine_inplace:da3nested-giant-large"
)

if [ -n "${DRIVES:-}" ]; then
    IFS=',' read -ra DRIVES_ARR <<< "$DRIVES"
else
    DRIVES_ARR=("${ALL_DRIVES[@]}")
fi

if [ -n "${EXPERIMENTS:-}" ]; then
    IFS=',' read -ra EXP_NAMES <<< "$EXPERIMENTS"
    EXPERIMENTS_ARR=()
    for want in "${EXP_NAMES[@]}"; do
        for tup in "${ALL_EXPERIMENTS[@]}"; do
            [ "${tup%%:*}" = "$want" ] && EXPERIMENTS_ARR+=("$tup")
        done
    done
else
    EXPERIMENTS_ARR=("${ALL_EXPERIMENTS[@]}")
fi

LOG_DIR="$PROJECT_ROOT/evals/logs/kitti"
GT_DIR="$PROJECT_ROOT/evals/logs/kitti_gt"
RESULTS_CSV="$PROJECT_ROOT/evals/logs/kitti_results.csv"
SWEEP_LOG="$PROJECT_ROOT/evals/logs/kitti_bigmodel.log"
mkdir -p "$LOG_DIR" "$GT_DIR"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "Method,Drive,ATE_RMSE,ScaleFactor,LC_Count,TUM_Poses,Depth_Frames,Status" > "$RESULTS_CSV"
fi

# GT TUM per drive
for drive in "${DRIVES_ARR[@]}"; do
    gt_tum="$GT_DIR/${drive}.tum"
    if [ ! -s "$gt_tum" ]; then
        date_prefix="${drive:0:10}"
        drive_dir="$KITTI_ROOT/$date_prefix/$drive"
        echo "[gt] Converting oxts → TUM for $drive"
        python evals/kitti_oxts_to_tum.py "$drive_dir" "$gt_tum"
    fi
done

run_one() {
    local exp="$1"; local method_key="$2"; local model="$3"; local drive="$4"

    echo ""
    echo "============================================================"
    echo "  $exp ($model)  on  $drive"
    echo "============================================================"

    local tum_path="$LOG_DIR/${exp}_${drive}.tum"
    local out_dir="$PROJECT_ROOT/docs/eval/kitti/$exp/$drive"

    XFORMERS_DISABLED=1 python evals/kitti_runner.py \
        --method "$method_key" \
        --experiment_name "$exp" \
        --drive "$drive" \
        --model_name "$model" \
        --kitti_root "$KITTI_ROOT" \
        --min_disparity "$MIN_DISPARITY" \
        --tum_path "$tum_path" --out_dir "$out_dir" \
        || { echo "  [SLAM FAIL]"; echo "$exp,$drive,,,,,,slam_fail" >> "$RESULTS_CSV"; return; }

    if [ ! -s "$tum_path" ]; then
        echo "  [empty TUM -- skipping eval]"
        echo "$exp,$drive,,,,,,empty_tum" >> "$RESULTS_CSV"
        return
    fi

    local gt_tum="$GT_DIR/${drive}.tum"
    local ape_out rmse scale
    ape_out=$(evo_ape tum "$gt_tum" "$tum_path" -as 2>&1 || true)
    rmse=$(echo "$ape_out"  | { grep -E "^\s*rmse" || true; } | head -1 | sed -E 's/.*rmse[^0-9.-]*([0-9.eE+-]+).*/\1/' || true)
    scale=$(echo "$ape_out" | { grep -Ei "scale"  || true; } | head -1 | sed -E 's/.*[^0-9.eE+-]([0-9.eE+-]+).*/\1/' || true)
    rmse="${rmse:-}"; scale="${scale:-}"

    local lc_count tum_poses depth_frames
    lc_count=$(cat "$out_dir/loop_closure_count.txt" 2>/dev/null | tr -d '[:space:]' || true)
    lc_count="${lc_count:-0}"
    tum_poses=$(wc -l < "$tum_path" | tr -d '[:space:]' || echo 0)
    depth_frames=$(ls "$out_dir" 2>/dev/null | { grep -c "^depth_frame_.*\.npy$" || true; })
    depth_frames="${depth_frames:-0}"

    echo "  ATE RMSE: ${rmse:---}  Scale: ${scale:---}  LC: $lc_count  Poses: $tum_poses  Depth: $depth_frames"
    echo "$exp,$drive,$rmse,$scale,$lc_count,$tum_poses,$depth_frames,ok" >> "$RESULTS_CSV"

    local depth_json="$LOG_DIR/${exp}_${drive}_depth.json"
    python evals/eval_kitti_depth.py \
        --method "$exp" --drive "$drive" \
        --pred_dir "$out_dir" --kitti_root "$KITTI_ROOT" \
        --output "$depth_json" \
        || echo "  [depth eval fail -- continuing]"
}

{
    for tup in "${EXPERIMENTS_ARR[@]}"; do
        IFS=':' read -r exp method_key model <<< "$tup"
        for drive in "${DRIVES_ARR[@]}"; do
            run_one "$exp" "$method_key" "$model" "$drive"
        done
    done
} 2>&1 | tee -a "$SWEEP_LOG"

# Dedupe CSV (keep last row per method,drive)
python - <<'PYEOF'
import csv, os
csv_path = "evals/logs/kitti_results.csv"
rows, header = [], None
with open(csv_path) as f:
    r = csv.reader(f); header = next(r); rows = list(r)
last = {}
for i, row in enumerate(rows):
    if len(row) < 2: continue
    last[(row[0], row[1])] = i
keep = sorted(last.values())
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(header)
    for i in keep: w.writerow(rows[i])
print(f"[dedupe] kept {len(keep)} rows")
PYEOF

echo ""
echo "============================================================"
echo "Big-model sweep done. CSV: $RESULTS_CSV   Log: $SWEEP_LOG"
echo "============================================================"
