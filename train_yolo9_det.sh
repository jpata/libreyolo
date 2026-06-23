#!/usr/bin/env bash
# =============================================================================
# train_yolo9_det.sh — Train YOLOv9 detection (7 thing classes)
#
# Usage:
#   ./train_yolo9_det.sh --output OUTPUT_DIR [--model-size SIZE] [--epochs N]
#                        [--batch N] [--imgsz N] [--device DEVICE] [--workers N]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="../data/processed/offsed_segdet"
MODEL_SIZE="s"
EPOCHS=100
BATCH=32
IMGSZ=416
DEVICE=""
WORKERS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
DATA_YAML="${OUTPUT_DIR}/dataset_yolo9_det.yaml"
RESULT_FILE="${OUTPUT_DIR}/result_yolo9_det.json"

if [ ! -f "$DATA_YAML" ]; then
    echo "ERROR: Dataset YAML not found: $DATA_YAML"
    echo "Run ./prepare_offsed.sh first."
    exit 1
fi

cd "$SCRIPT_DIR"

echo "============================================"
echo "  Train YOLOv9-${MODEL_SIZE} (detection)"
echo "============================================"
echo "  Data:     $DATA_YAML"
echo "  Epochs:   $EPOCHS"
echo "  Batch:    $BATCH"
echo "  Image:    $IMGSZ"
echo "  Device:   ${DEVICE:-auto}"
echo "============================================"
echo ""

uv run --no-sync python3 -c "
import json
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

from libreyolo import LibreYOLO9

model = LibreYOLO9(
    model_path=None,
    size='$MODEL_SIZE',
    nb_classes=7,
    task='detect',
    device='${DEVICE:-auto}',
)

results = model.train(
    data='$DATA_YAML',
    epochs=$EPOCHS,
    batch=$BATCH,
    imgsz=$IMGSZ,
    workers=$WORKERS,
    device='${DEVICE:-auto}',
    project='runs/yolo9_det',
    name='train',
    exist_ok=True,
)

summary = {
    'model': 'YOLOv9',
    'task': 'detect',
    'final_loss': results.get('final_loss'),
    'best_mAP50': results.get('best_mAP50'),
    'best_mAP50_95': results.get('best_mAP50_95'),
    'best_epoch': results.get('best_epoch'),
    'best_checkpoint': str(results.get('best_checkpoint', '')),
    'save_dir': str(results.get('save_dir', '')),
}
with open('$RESULT_FILE', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'  Final loss:     {summary[\"final_loss\"]:.4f}')
print(f'  Best mAP50:     {summary[\"best_mAP50\"]:.4f}')
print(f'  Best mAP50-95:  {summary[\"best_mAP50_95\"]:.4f}')
print(f'  Best epoch:     {summary[\"best_epoch\"]}')
"

echo ""
echo "YOLOv9 detection training complete. Results: $RESULT_FILE"
