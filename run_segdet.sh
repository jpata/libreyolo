#!/usr/bin/env bash
# =============================================================================
# run_segdet.sh — Preprocess Offsed dataset and train comparison models
#
# Trains three models on the same Offsed dataset:
#   1. LibreSegDet  (joint detection + semantic, using LibreSegDet architecture)
#   2. YOLOv9       (detection only)
#   3. YOLOv9       (semantic segmentation only)
#
# Validation performance is compared across all three at the end.
#
# Usage:
#   ./run_segdet.sh [--raw RAW_DIR] [--output OUTPUT_DIR] [--model-size SIZE]
#                   [--epochs N] [--batch N] [--imgsz N] [--device DEVICE]
#                   [--workers N]
#
# Examples:
#   # Full pipeline: preprocess + train three models size-s for 100 epochs
#   ./run_segdet.sh --raw data/raw/offsed --output data/processed/offsed_segdet \
#                   --model-size s --epochs 100 --batch 8
#
#   # Train only (skip preprocessing)
#   ./run_segdet.sh --skip-preprocess --model-size m --epochs 200 --batch 4
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_SCRIPT="$SCRIPT_DIR/weights/prepare_offsed_data.py"
DATA_YAML="$SCRIPT_DIR/libreyolo/config/datasets/offsed_segdet.yaml"

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
EXTRA_ARGS=()

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
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- Resolve paths ----
RAW_DIR="$(realpath -m "$RAW_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

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

# ---- Step 1: Preprocess Offsed dataset ----
if [ "$SKIP_PREPROCESS" = false ]; then
    echo "[Step 1] Preprocessing Offsed dataset..."
    echo "  Raw:    $RAW_DIR"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    if [ ! -d "$RAW_DIR" ]; then
        echo "ERROR: Raw data directory not found: $RAW_DIR"
        echo "Please download the Offsed dataset first or use --skip-preprocess."
        exit 1
    fi

    uv run --no-sync python3 "$PREPARE_SCRIPT" \
        --raw "$RAW_DIR" \
        --output "$OUTPUT_DIR"

    echo ""
    echo "[Step 1] Preprocessing complete."
    echo ""
else
    echo "[Step 1] Skipping preprocessing (--skip-preprocess)."
    echo ""
fi

# ---- Step 2: Create task-specific YAML configs ----
echo "[Step 2] Creating task-specific YAML configs..."

# 2a. LibreSegDet YAML (joint detection + semantic, nc=20, masks_dir)
DATA_YAML_TMP="${OUTPUT_DIR}/dataset_segdet.yaml"
cp "$DATA_YAML" "$DATA_YAML_TMP"
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s|^path:.*$|path: ${OUTPUT_DIR}|" "$DATA_YAML_TMP"
else
    sed -i "s|^path:.*$|path: ${OUTPUT_DIR}|" "$DATA_YAML_TMP"
fi

# 2b. YOLOv9 detection YAML (nc=7, thing classes only)
YOLO9_DET_YAML="${OUTPUT_DIR}/dataset_yolo9_det.yaml"
cat > "$YOLO9_DET_YAML" << YAMLEOF
# YOLOv9 detection config for Offsed (thing classes only)
path: ${OUTPUT_DIR}
train: images
val: images
nc: 7
names:
  0: obstacle
  1: person
  2: held object
  3: car
  4: truck
  5: camper
  6: excavator
YAMLEOF

# 2c. YOLOv9 semantic YAML (nc=20, with masks_dir)
YOLO9_SEM_YAML="${OUTPUT_DIR}/dataset_yolo9_sem.yaml"
cat > "$YOLO9_SEM_YAML" << YAMLEOF
# YOLOv9 semantic segmentation config for Offsed (all classes)
path: ${OUTPUT_DIR}
train: images
val: images
nc: 20
names:
  0: background
  1: obstacle
  2: person
  3: held object
  4: car
  5: truck
  6: camper
  7: excavator
  8: road
  9: sky
  10: building
  11: sidewalk
  12: tree
  13: grass
  14: water
  15: rock
  16: fence
  17: mountain
  18: void
  19: other
masks_dir: masks
YAMLEOF

echo "  LibreSegDet YAML:  $DATA_YAML_TMP"
echo "  YOLOv9 det YAML:   $YOLO9_DET_YAML"
echo "  YOLOv9 sem YAML:   $YOLO9_SEM_YAML"
echo ""

# ---- Step 3: Verify data ----
echo "[Step 3] Verifying data..."
IMG_COUNT=$(find "$OUTPUT_DIR/images" -type f 2>/dev/null | wc -l)
MASK_COUNT=$(find "$OUTPUT_DIR/masks" -type f 2>/dev/null | wc -l)
LABEL_COUNT=$(find "$OUTPUT_DIR/labels" -type f 2>/dev/null | wc -l)
echo "  Images: $IMG_COUNT"
echo "  Masks:  $MASK_COUNT"
echo "  Labels: $LABEL_COUNT"

if [ "$IMG_COUNT" -eq 0 ]; then
    echo "ERROR: No images found in $OUTPUT_DIR/images."
    echo "Run preprocessing first or check the output path."
    exit 1
fi
echo ""

# ---- Result files for comparison ----
SEGDET_RESULT="${OUTPUT_DIR}/result_segdet.json"
YOLO9_DET_RESULT="${OUTPUT_DIR}/result_yolo9_det.json"
YOLO9_SEM_RESULT="${OUTPUT_DIR}/result_yolo9_sem.json"

cd "$SCRIPT_DIR"

# ---- Step 4: Train LibreSegDet (joint detection + semantic) ----
echo "[Step 4/6] Training LibreSegDet-${MODEL_SIZE}..."
echo ""

uv run --no-sync python3 -c "
import json
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

from libreyolo.models.segdet.model import LibreSegDet

model = LibreSegDet(
    model_path=None,
    size='$MODEL_SIZE',
    nb_classes=20,
    num_thing_classes=7,
    task='detect',
    device='${DEVICE:-auto}',
)

results = model.train(
    data='$DATA_YAML_TMP',
    epochs=$EPOCHS,
    batch=$BATCH,
    imgsz=$IMGSZ,
    workers=$WORKERS,
    device='${DEVICE:-auto}',
    project='runs/segdet',
    name='train',
    exist_ok=True,
)

summary = {
    'model': 'LibreSegDet',
    'task': 'detect+semantic',
    'final_loss': results.get('final_loss'),
    'best_mAP50': results.get('best_mAP50'),
    'best_mAP50_95': results.get('best_mAP50_95'),
    'best_epoch': results.get('best_epoch'),
    'best_checkpoint': str(results.get('best_checkpoint', '')),
    'save_dir': str(results.get('save_dir', '')),
}
with open('$SEGDET_RESULT', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'  Final loss:  {summary[\"final_loss\"]:.4f}')
print(f'  Best mAP50:  {summary[\"best_mAP50\"]:.4f}')
print(f'  Best mAP50-95: {summary[\"best_mAP50_95\"]:.4f}')
print(f'  Best epoch:  {summary[\"best_epoch\"]}')
"

echo ""
echo "[Step 4/6] LibreSegDet training complete."
echo ""

# ---- Step 5: Train YOLOv9 detection ----
echo "[Step 5/6] Training YOLOv9-${MODEL_SIZE} (detection)..."
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
    data='$YOLO9_DET_YAML',
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
with open('$YOLO9_DET_RESULT', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'  Final loss:  {summary[\"final_loss\"]:.4f}')
print(f'  Best mAP50:  {summary[\"best_mAP50\"]:.4f}')
print(f'  Best mAP50-95: {summary[\"best_mAP50_95\"]:.4f}')
print(f'  Best epoch:  {summary[\"best_epoch\"]}')
"

echo ""
echo "[Step 5/6] YOLOv9 detection training complete."
echo ""

# ---- Step 6: Train YOLOv9 semantic ----
echo "[Step 6/6] Training YOLOv9-${MODEL_SIZE} (semantic segmentation)..."
echo ""

uv run --no-sync python3 -c "
import json
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

from libreyolo import LibreYOLO9

model = LibreYOLO9(
    model_path=None,
    size='$MODEL_SIZE',
    nb_classes=20,
    task='semantic',
    device='${DEVICE:-auto}',
)

results = model.train(
    data='$YOLO9_SEM_YAML',
    epochs=$EPOCHS,
    batch=$BATCH,
    imgsz=$IMGSZ,
    workers=$WORKERS,
    device='${DEVICE:-auto}',
    project='runs/yolo9_sem',
    name='train',
    exist_ok=True,
)

# For semantic, best_mAP50_95 stores mIoU
summary = {
    'model': 'YOLOv9',
    'task': 'semantic',
    'final_loss': results.get('final_loss'),
    'best_mIoU': results.get('best_mAP50_95'),
    'best_epoch': results.get('best_epoch'),
    'best_checkpoint': str(results.get('best_checkpoint', '')),
    'save_dir': str(results.get('save_dir', '')),
}
with open('$YOLO9_SEM_RESULT', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'  Final loss:  {summary[\"final_loss\"]:.4f}')
print(f'  Best mIoU:   {summary[\"best_mIoU\"]:.4f}')
print(f'  Best epoch:  {summary[\"best_epoch\"]}')
"

echo ""
echo "[Step 6/6] YOLOv9 semantic training complete."
echo ""

# ---- Step 7: Comparison summary ----
echo "============================================"
echo "  Performance Comparison"
echo "============================================"

uv run --no-sync python3 -c "
import json

results = {}
for label, path in [
    ('LibreSegDet (joint)', '$SEGDET_RESULT'),
    ('YOLOv9 (detect)',     '$YOLO9_DET_RESULT'),
    ('YOLOv9 (semantic)',   '$YOLO9_SEM_RESULT'),
]:
    try:
        with open(path) as f:
            results[label] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results[label] = {'error': str(e)}

print()
print(f'{\"Model\":24s} {\"Task\":18s} {\"mAP50\":>8s} {\"mAP50-95\":>10s} {\"mIoU\":>8s} {\"Best Ep\":>8s}')
print('-' * 78)

for label in ['LibreSegDet (joint)', 'YOLOv9 (detect)', 'YOLOv9 (semantic)']:
    r = results.get(label, {})
    if 'error' in r:
        print(f'{label:24s}  ERROR: {r[\"error\"]}')
        continue

    task    = r.get('task', '?')
    mAP50   = r.get('best_mAP50')
    mAP95   = r.get('best_mAP50_95')
    miou    = r.get('best_mIoU')
    epoch   = r.get('best_epoch')

    mAP50_str = f'{mAP50:.4f}' if mAP50 is not None else '  N/A  '
    mAP95_str = f'{mAP95:.4f}' if mAP95 is not None else '  N/A  '
    miou_str  = f'{miou:.4f}'  if miou  is not None else '  N/A  '
    epoch_str = f'{epoch}'     if epoch is not None else '  N/A  '

    print(f'{label:24s} {task:18s} {mAP50_str:>8s} {mAP95_str:>10s} {miou_str:>8s} {epoch_str:>8s}')

print()
print('Notes:')
print('  - LibreSegDet trains both detection and semantic heads jointly.')
print('    Validation only reports detection metrics (mAP50/mAP50-95).')
print('  - YOLOv9 (detect) trains detection only on 7 thing classes.')
print('  - YOLOv9 (semantic) trains semantic segmentation on all 20 classes.')
print('    Its mIoU is shown in the mIoU column.')
"
