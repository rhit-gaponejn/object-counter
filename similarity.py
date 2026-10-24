"""CLIP-based similarity matching for the 'click one example object' flow.

Given a clicked example crop, find which detected objects are visually similar.
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        _processor = CLIPProcessor.from_pretrained(MODEL_ID)
        _model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
        _model.eval()
    return _processor, _model


def _crop(image: Image.Image, box) -> Image.Image:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    return image.crop((x1, y1, x2, y2))


@torch.no_grad()
def _embed(crops: list[Image.Image]) -> torch.Tensor:
    processor, model = _load()
    inputs = processor(images=crops, return_tensors="pt").to(DEVICE)
    feats = model.get_image_features(**inputs)
    if not torch.is_tensor(feats):
        feats = feats.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def match_by_example(image: Image.Image, detections: list[dict], example_idx: int,
                     similarity_threshold: float = 0.75):
    """Compare every detection crop to the clicked example via CLIP embeddings.

    Returns (matches, scores) where matches is a list of detection indices that
    pass the similarity threshold, and scores maps index -> cosine similarity.
    """
    if not detections or example_idx >= len(detections):
        return [], {}

    crops = [_crop(image, d["box"]) for d in detections]
    # Skip crops too small to be meaningful for CLIP
    valid = [i for i, c in enumerate(crops) if c.width >= 5 and c.height >= 5]
    if example_idx not in valid:
        return [], {}

    feats = _embed([crops[i] for i in valid]).cpu()
    example = feats[valid.index(example_idx)]

    scores = {}
    matches = []
    for j, i in enumerate(valid):
        sim = float((feats[j] @ example).item())
        scores[i] = sim
        if i == example_idx or sim >= similarity_threshold:
            matches.append(i)
    return matches, scores
