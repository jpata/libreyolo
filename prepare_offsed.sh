#!/usr/bin/env bash
# =============================================================================
# prepare_offsed.sh — Preprocess Offsed dataset and create task-specific YAMLs
#
# Converts raw Offsed data into the processed format with:
#   - images/       resized images
#   - labels/       YOLO-format box labels
#   - masks/        semantic segmentation masks
#   - *.yaml        per-task dataset configs
#
# Usage:
#   ./prepare_offsed.sh --raw RAW_DIR --output OUTPUT_DIR
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_SCRIPT="$SCRIPT_DIR/weights/prepare_offsed_data.py"
DATA_YAML="$SCRIPT_DIR/libreyolo/config/datasets/offsed_segdet.yaml"

RAW_DIR="../data/raw/offsed"
OUTPUT_DIR="../data/processed/offsed_segdet"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw) RAW_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RAW_DIR="$(realpath -m "$RAW_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

echo "============================================"
echo "  Prepare Offsed Dataset"
echo "============================================"
echo "  Raw:    $RAW_DIR"
echo "  Output: $OUTPUT_DIR"
echo "============================================"
echo ""

if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: Raw data directory not found: $RAW_DIR"
    exit 1
fi

# Step 1: Run preprocessing
uv run --no-sync python3 "$PREPARE_SCRIPT" \
    --raw "$RAW_DIR" \
    --output "$OUTPUT_DIR"

# Step 2: Create task-specific YAML configs
echo ""
echo "Creating task-specific YAML configs..."

# 2a. LibreSegDet YAML (joint detection + semantic, nc=20)
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

# Step 3: Verify data
echo ""
echo "Verifying data..."
IMG_COUNT=$(find "$OUTPUT_DIR/images" -type f 2>/dev/null | wc -l)
MASK_COUNT=$(find "$OUTPUT_DIR/masks" -type f 2>/dev/null | wc -l)
LABEL_COUNT=$(find "$OUTPUT_DIR/labels" -type f 2>/dev/null | wc -l)
echo "  Images: $IMG_COUNT"
echo "  Masks:  $MASK_COUNT"
echo "  Labels: $LABEL_COUNT"

if [ "$IMG_COUNT" -eq 0 ]; then
    echo "ERROR: No images found in $OUTPUT_DIR/images."
    exit 1
fi

echo ""
echo "Preprocessing complete."
