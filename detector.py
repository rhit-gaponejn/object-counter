"""Grounding DINO wrapper: open-vocabulary object detection via text prompt.

Runs locally on GPU (falls back to CPU). Free, open-weight model from Hugging Face.
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
        _model.eval()
    return _processor, _model


def detect(image: Image.Image, text_prompt: str, box_threshold: float = 0.3,
           text_threshold: float = 0.25):
    """Detect all objects matching text_prompt.

    Returns list of dicts: {"box": [x1, y1, x2, y2], "score": float, "label": str}
    Box coordinates are absolute pixels in the original image size.
    """
    processor, model = _load()
    prompt = text_prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (H, W)
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        # We repeat the prompt for dense scenes; if the phrase came back doubled
        # (e.g. "coin coin"), collapse it back to "coin".
        words = label.split()
        if len(words) % 2 == 0 and words[:len(words) // 2] == words[len(words) // 2:]:
            label = " ".join(words[:len(words) // 2])
        detections.append({
            "box": [float(v) for v in box.tolist()],
            "score": float(score),
            "label": label,
        })

    # Deduplicate overlapping boxes (dense scenes produce many duplicates).
    return nms(detections)


# ------------------------------------------------------------------ NMS

def _iou(a, b):
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(a, b):
    """Fraction of box a's area that lies inside box b."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    return inter / area_a if area_a > 0 else 0.0


def nms(detections: list[dict], iou_threshold: float = 0.5,
        containment_threshold: float = 0.8) -> list[dict]:
    """Remove duplicate/overlapping boxes.

    Grounding DINO often emits several boxes for the same object in dense
    scenes. Keep the highest-scoring box and drop boxes that overlap it
    heavily (high IoU) or that are almost entirely contained in it.
    """
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: list[dict] = []
    for d in dets:
        duplicate = False
        for k in kept:
            if _iou(d["box"], k["box"]) >= iou_threshold:
                duplicate = True
                break
            if (_containment(d["box"], k["box"]) >= containment_threshold
                    or _containment(k["box"], d["box"]) >= containment_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    # Restore discovery order so numbering reads left-to-right/top-to-bottom.
    kept.sort(key=lambda d: (d["box"][1], d["box"][0]))
    return kept
