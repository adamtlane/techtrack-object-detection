"""Evaluate the TechTrack YOLO models on labeled frames with standard detection metrics.

Usage (from the techtrack folder):
    python scripts/evaluate_models.py [storage/logistics] [storage/yolo_models]

For each model it runs the Detector over every labeled image, applies TechTrack's NMS,
and reports:
  - mAP@0.5: per class, detections are sorted by score and matched greedily to unmatched
    ground truth boxes of the same class at IoU >= 0.5. AP is the all point interpolated
    area under the precision recall curve, averaged over classes with ground truth.
  - precision, recall, and F1 at the service's default score threshold (0.5) and at the
    threshold that maximizes F1.
Labels are YOLO text files (class cx cy w h, normalized) next to each image.
"""
import glob, json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "modules"))
from inference.model import Detector
from inference.nms import NMS

DATA = sys.argv[1] if len(sys.argv) > 1 else "storage/logistics"
MODELS = sys.argv[2] if len(sys.argv) > 2 else "storage/yolo_models"
IOU_MATCH, NMS_IOU, SCORE_FLOOR = 0.5, 0.4, 0.01


def decode(outputs, W, H):
    """Same decoding as Detector.post_process, with a low score floor for complete PR curves."""
    boxes, ids, obj, score = [], [], [], []
    for out in outputs:
        for d in out:
            cx, cy, w, h = d[:4]; o = d[4]; c = int(np.argmax(d[5:])); s = o * d[5 + c]
            if o > SCORE_FLOOR and s > SCORE_FLOOR:
                boxes.append([int((cx - w / 2) * W), int((cy - h / 2) * H), int(w * W), int(h * H)])
                ids.append(c); obj.append(float(o)); score.append(float(s))
    return boxes, ids, obj, score


def ground_truth(img_path, W, H):
    gt = []
    for line in open(os.path.splitext(img_path)[0] + ".txt"):
        p = line.split()
        if len(p) >= 5:
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            gt.append((c, [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]))
    return gt


def iou(a, b):
    xl, yt, xr, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if xr <= xl or yb <= yt:
        return 0.0
    inter = (xr - xl) * (yb - yt)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(dets, gts, num_classes):
    """Return per class lists of (score, is_tp) and ground truth counts."""
    per_class, n_gt = {}, {}
    for c in range(num_classes):
        recs = sorted(((s, i, b) for i, d in enumerate(dets) for (k, s, b) in d if k == c), key=lambda r: -r[0])
        gt = {i: [b for (k, b) in g if k == c] for i, g in enumerate(gts)}
        used = {i: np.zeros(len(v), bool) for i, v in gt.items()}
        n_gt[c] = sum(len(v) for v in gt.values())
        out = []
        for s, i, b in recs:
            best, bi = 0.0, -1
            for g, gb in enumerate(gt[i]):
                o = iou(b, gb)
                if o > best and not used[i][g]:
                    best, bi = o, g
            tp = best >= IOU_MATCH
            if tp:
                used[i][bi] = True
            out.append((s, tp))
        per_class[c] = out
    return per_class, n_gt


def average_precision(recs, n):
    if n == 0:
        return None
    tp = np.cumsum([t for _, t in recs]); fp = np.cumsum([not t for _, t in recs])
    rec = tp / n if len(recs) else np.array([]); prec = tp / np.maximum(tp + fp, 1e-9) if len(recs) else np.array([])
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def operating_point(per_class, n_gt, thr):
    tp = sum(t for recs in per_class.values() for s, t in recs if s >= thr)
    fp = sum(not t for recs in per_class.values() for s, t in recs if s >= thr)
    total = sum(n_gt.values()); p = tp / max(tp + fp, 1); r = tp / max(total, 1)
    return {"threshold": round(thr, 2), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def evaluate(model_dir, images):
    weights = glob.glob(os.path.join(model_dir, "*.weights"))[0]
    cfg = glob.glob(os.path.join(model_dir, "*.cfg"))[0]
    names = os.path.join(model_dir, "logistics.names")
    classes = [l.strip() for l in open(names) if l.strip()]
    det, nms = Detector(weights, cfg, names), NMS(score_threshold=SCORE_FLOOR, nms_iou_threshold=NMS_IOU)
    dets, gts = [], []
    for p in images:
        img = cv2.imread(p); H, W = img.shape[:2]
        b, i, o, s = decode(det.predict(img), W, H)
        fb, fi, _, fs = nms.filter(b, i, o, s) if b else ([], [], [], [])
        dets.append([(k, sc, [x, y, x + w, y + h]) for (x, y, w, h), k, sc in zip(fb, fi, fs)])
        gts.append(ground_truth(p, W, H))
    per_class, n_gt = match(dets, gts, len(classes))
    aps = {classes[c]: average_precision(per_class[c], n_gt[c]) for c in range(len(classes)) if n_gt[c]}
    sweep = [operating_point(per_class, n_gt, t) for t in np.arange(0.05, 0.96, 0.05)]
    return {"mAP50": round(float(np.mean(list(aps.values()))), 4),
            "default_threshold": operating_point(per_class, n_gt, 0.5),
            "best_f1_threshold": max(sweep, key=lambda r: r["f1"]),
            "per_class_ap": {k: round(v, 4) for k, v in aps.items()}}


if __name__ == "__main__":
    images = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(DATA, f"*.{ext}"))
                    if os.path.exists(os.path.splitext(p)[0] + ".txt"))
    results = {"images": len(images)}
    for model_dir in sorted(glob.glob(os.path.join(MODELS, "yolo_model_*"))):
        results[os.path.basename(model_dir)] = r = evaluate(model_dir, images)
        print(os.path.basename(model_dir), {k: v for k, v in r.items() if k != "per_class_ap"}, flush=True)
    json.dump(results, open("evaluation_results.json", "w"), indent=2)
