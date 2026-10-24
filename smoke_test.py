"""Smoke test: generate a synthetic image with 6 'coins', detect & count them."""
from PIL import Image, ImageDraw, ImageFont
import detector, similarity, segmenter

img = Image.new("RGB", (640, 480), (90, 90, 95))
d = ImageDraw.Draw(img)
coords = [(80, 100), (300, 120), (500, 90), (150, 320), (380, 330), (550, 300)]
for (cx, cy) in coords:
    d.ellipse([cx - 35, cy - 35, cx + 35, cy + 35], fill=(200, 170, 60), outline=(150, 120, 30), width=3)
    d.text((cx - 8, cy - 10), "$", fill=(80, 60, 10))

dets = detector.detect(img, "coin . coin .", box_threshold=0.3)
print(f"detections: {len(dets)}")
for x in dets:
    print(f"  {x['label']}  score={x['score']:.2f}  box={[round(v) for v in x['box']]}")

assert len(dets) == 6, f"expected 6, got {len(dets)}"

matches, scores = similarity.match_by_example(img, dets, 0, similarity_threshold=0.75)
print(f"CLIP matches for example #1: {len(matches)} -> {[f'{scores[m]:.2f}' for m in matches]}")
assert 0 in matches

masks = segmenter.segment_boxes(img, [x["box"] for x in dets])
print(f"SAM masks: {len(masks)}")
assert len(masks) == 6

# NMS: duplicate/overlapping boxes must be deduplicated
dupes = [
    {"box": [45, 60, 116, 139], "score": 0.4, "label": "coin"},   # overlaps det #1
    {"box": [0, 0, 640, 480], "score": 0.35, "label": "coin"},    # contains everything
]
deduped = detector.nms(dets + dupes)
print(f"NMS: {len(dets) + len(dupes)} boxes -> {len(deduped)} after dedup")
assert len(deduped) == 6, f"NMS failed: {len(deduped)}"

# Click-point mapping: clicking inside a box must find it
import app
app._state.update(detections=deduped, image=img, prompt="coin")
for cx, cy in coords:
    i = app._detection_at_point(cx, cy, deduped)
    assert i is not None, f"click at ({cx},{cy}) found no box"
print("click-point mapping: OK")

print("ALL TESTS PASSED")
