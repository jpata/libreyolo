#!/usr/bin/env bash
# =============================================================================
# run_segdet.sh — Orchestrator: prepare data, train 3 models, compare results
#
# Delegates to:
#   prepare_offsed.sh   — preprocess Offsed dataset + create YAML configs
#   train_segdet.sh     — train LibreSegDet (joint detection + semantic)
#   train_yolo9_det.sh  — train YOLOv9 detection only
#   train_yolo9_sem.sh  — train YOLOv9 semantic segmentation only
#   compare_segdet.sh   — print comparison table
#
# Usage:
#   ./run_segdet.sh [--raw RAW_DIR] [--output OUTPUT_DIR] [--model-size SIZE]
#                   [--epochs N] [--batch N] [--imgsz N] [--device DEVICE]
#                   [--workers N] [--skip-preprocess]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults ----
RAW_DIR="../data/raw/offsed"
OUTPUT_DIR="../data/processed/offsed_segdet"
MODEL_SIZE="s"
EPOCHS=100
BATCH=32
IMGSZ=416
DEVICE=""
WORKERS=4
SKIP_PREPROCESS=false

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw) RAW_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Build common args ----
TRAIN_ARGS=(
    --output "$OUTPUT_DIR"
    --model-size "$MODEL_SIZE"
    --epochs "$EPOCHS"
    --batch "$BATCH"
    --imgsz "$IMGSZ"
    --workers "$WORKERS"
)
if [ -n "$DEVICE" ]; then
    TRAIN_ARGS+=(--device "$DEVICE")
fi

echo "============================================"
echo "  Offsed Model Comparison Pipeline"
echo "============================================"
echo "  Raw data:       $RAW_DIR"
echo "  Output:         $OUTPUT_DIR"
echo "  Model size:     $MODEL_SIZE"
echo "  Epochs:         $EPOCHS"
echo "  Batch size:     $BATCH"
echo "  Image size:     $IMGSZ"
echo "  Workers:        $WORKERS"
echo "  Device:         ${DEVICE:-auto}"
echo "  Skip preproc:   $SKIP_PREPROCESS"
echo "============================================"
echo ""

# ---- Step 1: Prepare ----
if [ "$SKIP_PREPROCESS" = false ]; then
    "$SCRIPT_DIR/prepare_offsed.sh" --raw "$RAW_DIR" --output "$OUTPUT_DIR"
else
    echo "[Step 1] Skipping preprocessing (--skip-preprocess)."
    echo ""
fi

# ---- Step 2-4: Train ----
echo "[Step 2/4] Training LibreSegDet..."
echo ""
"$SCRIPT_DIR/train_segdet.sh" "${TRAIN_ARGS[@]}"

echo ""
echo "[Step 3/4] Training YOLOv9 (detection)..."
echo ""
"$SCRIPT_DIR/train_yolo9_det.sh" "${TRAIN_ARGS[@]}"

echo ""
echo "[Step 4/4] Training YOLOv9 (semantic)..."
echo ""
"$SCRIPT_DIR/train_yolo9_sem.sh" "${TRAIN_ARGS[@]}"

# ---- Step 5: Compare ----
echo ""
"$SCRIPT_DIR/compare_segdet.sh" --output "$OUTPUT_DIR"
