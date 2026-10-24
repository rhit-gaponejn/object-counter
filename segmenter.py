"""SAM (Segment Anything) wrapper for pretty per-object mask annotations."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import SamProcessor, SamModel

MODEL_ID = "facebook/sam-vit-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        _processor = SamProcessor.from_pretrained(MODEL_ID)
        _model = SamModel.from_pretrained(MODEL_ID).to(DEVICE)
        _model.eval()
    return _processor, _model


def segment_boxes(image: Image.Image, boxes: list[list[float]]):
    """Given bounding boxes, return per-box boolean masks (H, W numpy arrays).

    Returns list of masks; empty list if no boxes.
    """
    if not boxes:
        return []
    processor, model = _load()
    # SamProcessor expects input_boxes shaped (batch_size, num_boxes, 4)
    input_boxes = [[[float(v) for v in b] for b in boxes]]

    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]  # (num_boxes, 1, H, W)

    results = []
    for i in range(len(boxes)):
        m = masks[i, 0].numpy().astype(bool)
        if m.any():
            results.append(m)
    return results
