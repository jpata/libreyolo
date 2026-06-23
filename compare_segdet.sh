#!/usr/bin/env bash
# =============================================================================
# compare_segdet.sh — Print comparison table from training result JSONs
#
# Usage:
#   ./compare_segdet.sh --output OUTPUT_DIR
# =============================================================================

set -euo pipefail

OUTPUT_DIR="../data/processed/offsed_segdet"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

SEGDET_RESULT="${OUTPUT_DIR}/result_segdet.json"
YOLO9_DET_RESULT="${OUTPUT_DIR}/result_yolo9_det.json"
YOLO9_SEM_RESULT="${OUTPUT_DIR}/result_yolo9_sem.json"

echo "============================================"
echo "  Performance Comparison"
echo "============================================"
echo "  Output dir: $OUTPUT_DIR"
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
