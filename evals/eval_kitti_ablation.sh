#!/bin/bash
# KITTI ablation sweep driver.
#
# Runs:
#   Reference reruns : vggt_slam, da3_baseline, da3_large_refine
#   Scale method     : da3_baseline_scale_ransac, da3_baseline_scale_weighted
#   Chunk/overlap    : da3_baseline_chunk32_ov4, da3_baseline_chunk60_ov8
#   Keyframe disp.   : da3_baseline_mindisp10, da3_baseline_mindisp25
#
# Each experiment runs across all 13 KITTI val drives and appends
# (experiment_name, drive, ATE_RMSE, ...) rows to kitti_results.csv.
#
# Usage:
#   ./evals/eval_kitti_ablation.sh                    # run everything
#   SWEEP_GROUPS=scale ./evals/eval_kitti_ablation.sh       # run only scale sweep
#   SWEEP_GROUPS=rerun,scale,chunk,disp ./evals/eval_kitti_ablation.sh
#   DRIVES=2011_09_26_drive_0002_sync ./evals/eval_kitti_ablation.sh
#
# Groups: rerun | scale | chunk | disp | all (default)
# NOTE: $GROUPS is a bash builtin array (user GIDs); we use $SWEEP_GROUPS instead.

set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

KITTI_ROOT="${KITTI_ROOT:-/nfs/turbo/coe-jungaocv/siyuanb/github/Depth-Anything-3/data/cut3r_data/kitti_backup/kitti}"
MODEL_NAME="${MODEL_NAME:-da3-small}"

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

if [ -n "${DRIVES:-}" ]; then
    IFS=',' read -ra DRIVES_ARR <<< "$DRIVES"
else
    DRIVES_ARR=("${ALL_DRIVES[@]}")
fi

SWEEP_GROUPS="${SWEEP_GROUPS:-all}"

LOG_DIR="$PROJECT_ROOT/evals/logs/kitti"
GT_DIR="$PROJECT_ROOT/evals/logs/kitti_gt"
RESULTS_CSV="$PROJECT_ROOT/evals/logs/kitti_results.csv"
SWEEP_LOG="$PROJECT_ROOT/evals/logs/kitti_ablation.log"
mkdir -p "$LOG_DIR" "$GT_DIR"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "Method,Drive,ATE_RMSE,ScaleFactor,LC_Count,TUM_Poses,Depth_Frames,Status" > "$RESULTS_CSV"
fi

# Ensure GT TUM exists for every drive
for drive in "${DRIVES_ARR[@]}"; do
    gt_tum="$GT_DIR/${drive}.tum"
    if [ ! -s "$gt_tum" ]; then
        date_prefix="${drive:0:10}"
        drive_dir="$KITTI_ROOT/$date_prefix/$drive"
        echo "[gt] Converting oxts → TUM for $drive"
        python evals/kitti_oxts_to_tum.py "$drive_dir" "$gt_tum"
    fi
done

# ---- Helper: run one (experiment, drive) pair, log CSV row ----
# Usage: run_one EXPERIMENT_NAME METHOD_KEY MIN_DISP EXTRA_ARGS...
run_one() {
    local exp="$1"; shift
    local method_key="$1"; shift
    local min_disp="$1"; shift
    local extra_args=("$@")
    local drive="$CUR_DRIVE"

    echo ""
    echo "============================================================"
    echo "  $exp  on  $drive"
    echo "============================================================"

    local tum_path="$LOG_DIR/${exp}_${drive}.tum"
    local out_dir="$PROJECT_ROOT/docs/eval/kitti/$exp/$drive"

    XFORMERS_DISABLED=1 python evals/kitti_runner.py \
        --method "$method_key" \
        --experiment_name "$exp" \
        --drive "$drive" \
        --model_name "$MODEL_NAME" \
        --kitti_root "$KITTI_ROOT" \
        --min_disparity "$min_disp" \
        --tum_path "$tum_path" --out_dir "$out_dir" \
        --force \
        "${extra_args[@]}" \
        || { echo "  [SLAM FAIL]"; echo "$exp,$drive,,,,,,slam_fail" >> "$RESULTS_CSV"; return; }

    if [ ! -s "$tum_path" ]; then
        echo "  [empty TUM -- skipping eval]"
        echo "$exp,$drive,,,,,,empty_tum" >> "$RESULTS_CSV"
        return
    fi

    local gt_tum="$GT_DIR/${drive}.tum"
    local ape_out
    ape_out=$(evo_ape tum "$gt_tum" "$tum_path" -as 2>&1 || true)
    local rmse
    rmse=$(echo "$ape_out" | { grep -E "^\s*rmse" || true; } | head -1 | sed -E 's/.*rmse[^0-9.-]*([0-9.eE+-]+).*/\1/' || true)
    local scale
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

# Drop previous rows for an experiment name to keep the CSV clean after a rerun.
strip_csv_rows() {
    local exp="$1"
    local tmp
    tmp=$(mktemp)
    grep -v "^${exp}," "$RESULTS_CSV" > "$tmp" || true
    mv "$tmp" "$RESULTS_CSV"
}

want_group() {
    local g="$1"
    [ "$SWEEP_GROUPS" = "all" ] && return 0
    case ",$SWEEP_GROUPS," in
        *,$g,*) return 0 ;;
        *)      return 1 ;;
    esac
}

# ---- Main sweep ----
{
    for drive in "${DRIVES_ARR[@]}"; do
        CUR_DRIVE="$drive"

        if want_group "rerun"; then
            run_one "vggt_slam"        "vggt_slam"    15
            run_one "da3_baseline"     "da3_baseline" 15
            run_one "da3_large_refine" "da3_large_refine" 15
        fi

        if want_group "scale"; then
            run_one "da3_baseline_scale_ransac"  "da3_baseline" 15 --scale_method depth-ransac
            run_one "da3_baseline_scale_weighted" "da3_baseline" 15 --scale_method depth-weighted
        fi

        if want_group "chunk"; then
            run_one "da3_baseline_chunk32_ov4" "da3_baseline" 15 --chunk_size 32 --overlap 4
            run_one "da3_baseline_chunk60_ov8" "da3_baseline" 15 --chunk_size 60 --overlap 8
        fi

        if want_group "disp"; then
            run_one "da3_baseline_mindisp10" "da3_baseline" 10
            run_one "da3_baseline_mindisp25" "da3_baseline" 25
        fi
    done
} 2>&1 | tee -a "$SWEEP_LOG"

# If we rewrote reference rows, dedupe CSV -- keep the latest row per (method, drive) pair.
python - <<'PYEOF'
import csv, os
csv_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "evals/logs/kitti_results.csv")
rows, header = [], None
with open(csv_path) as f:
    r = csv.reader(f); header = next(r); rows = list(r)
# keep last occurrence of each (method, drive)
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
echo "Sweep complete. CSV: $RESULTS_CSV"
echo "Full log: $SWEEP_LOG"
echo "============================================================"
