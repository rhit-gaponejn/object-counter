"""Object counting app — count objects in an image by typing a name or clicking one example.

Run:  double-click run_app.bat  (or)  python app.py
Then open http://127.0.0.1:7860 in your browser.

Everything runs locally on your own hardware with free, open-weight models:
  - Grounding DINO (open-vocabulary detection from a text prompt)
  - CLIP (find objects visually similar to a clicked example)
  - SAM (optional segmentation masks for annotation)
"""
from __future__ import annotations

# Make the app runnable from ANY working directory (e.g. `python C:\...\app.py`
# from another folder, a shortcut, or VS Code) by ensuring this file's folder
# is on the module search path before importing detector/similarity/segmenter.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

import detector
import segmenter
import similarity

try:
    FONT = ImageFont.truetype("arial.ttf", 22)
except OSError:
    FONT = ImageFont.load_default()

# Keep the latest detection results in memory so the "click example" flow can reuse them.
_state: dict = {"detections": [], "image": None, "prompt": ""}


# ---------------------------------------------------------------- visualization

_PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128), (220, 190, 255), (255, 0, 128),
]


def draw_annotations(image: Image.Image, detections: list[dict],
                     masks=None, clicked_idx: int | None = None) -> Image.Image:
    img = image.convert("RGB")
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    odraw = ImageDraw.Draw(out)

    for i, det in enumerate(detections):
        color = _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = det["box"]
        if masks is not None and i < len(masks):
            m = masks[i]
            tint = np.zeros((*m.shape, 4), dtype=np.uint8)
            tint[m] = (*color, 90)
            out = Image.alpha_composite(out, Image.fromarray(tint))
            odraw = ImageDraw.Draw(out)
        width = 5 if i == clicked_idx else 3
        od.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=width)
        label = f"#{i + 1}"
        tw = od.textbbox((0, 0), label, font=FONT)
        ly1 = y1 - (tw[3] - tw[1] + 8) if y1 - (tw[3] - tw[1] + 8) >= 0 else y1
        od.rectangle([x1, ly1, x1 + (tw[2] - tw[0]) + 10, ly1 + (tw[3] - tw[1]) + 8],
                     fill=(*color, 255))
        odraw.text((x1 + 5, ly1 + 4), label, fill=(255, 255, 255, 255), font=FONT)

    return Image.alpha_composite(out, overlay).convert("RGB")


# ------------------------------------------------------------------- workflows

def _run_detection(image: Image.Image, prompt: str, box_threshold: float):
    text = prompt.strip() or "object"
    single = text.rstrip(". ")
    # Repeat the prompt to help the model pick up every instance in dense scenes.
    dets = detector.detect(image, f"{single} . {single} .", box_threshold=box_threshold)
    return dets, single


def count_by_text(image, prompt, box_threshold, show_masks):
    """Mode A: type an object name, get the count."""
    if image is None:
        raise gr.Error("Please load an image first.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    img = img.convert("RGB")

    dets, single = _run_detection(img, prompt, box_threshold)
    _state.update(detections=dets, image=img, prompt=single)

    masks = segmenter.segment_boxes(img, [d["box"] for d in dets]) if show_masks else None
    annotated = draw_annotations(img, dets, masks=masks)

    table = "\n".join(
        f"#{i + 1}  {d['label']}  score={d['score']:.2f}  box=[{d['box'][0]:.0f},"
        f"{d['box'][1]:.0f}, {d['box'][2]:.0f}, {d['box'][3]:.0f}]"
        for i, d in enumerate(dets)
    ) or "(no detections — try lowering the confidence threshold)"
    return annotated, f"**Count: {len(dets)}**", table


def _detection_at_point(x: float, y: float, detections: list[dict]) -> int | None:
    """Find the detection whose box contains (x, y); ties broken by score/area."""
    best, best_key = None, None
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        if x1 - 5 <= x <= x2 + 5 and y1 - 5 <= y <= y2 + 5:
            area = max(1.0, (x2 - x1) * (y2 - y1))
            key = (d["score"], -area)  # prefer higher score, then smaller box
            if best is None or key > best_key:
                best, best_key = i, key
    return best


def count_by_click(prompt, box_threshold, sim_threshold, show_masks, evt: gr.SelectData):
    """Mode B: user clicked an example — count visually similar objects via CLIP."""
    if _state.get("image") is None or not _state.get("detections"):
        raise gr.Error("Run 'Detect & Count' first (load image + prompt), then click an object.")

    dets = _state["detections"]

    # Gradio's Image.select returns the clicked pixel: index == [x, y].
    pos = evt.index
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        idx = _detection_at_point(float(pos[0]), float(pos[1]), dets)
    else:
        idx = int(pos) if pos is not None else None
    if idx is None:
        raise gr.Error(
            "No detection box under your click — click directly on one of the "
            "numbered boxes (or re-run Detect & Count with a lower threshold).")

    matches, scores = similarity.match_by_example(
        _state["image"], dets, idx, similarity_threshold=sim_threshold)

    filtered = [dets[i] for i in sorted(matches)]
    masks = (segmenter.segment_boxes(_state["image"], [d["box"] for d in filtered])
             if show_masks else None)
    annotated = draw_annotations(_state["image"], filtered, masks=masks)

    table = "\n".join(
        f"#{j + 1}  det_score={dets[i]['score']:.2f}  clip_sim={scores.get(i, 1.0):.2f}"
        for j, i in enumerate(sorted(matches))
    ) or "(no similar objects — lower the similarity threshold)"
    return annotated, f"**Count: {len(matches)}**  (clicked #{idx + 1} as example)", table


# ------------------------------------------------------------------------ UI

def build_ui():
    with gr.Blocks(title="Object Counter") as demo:
        gr.Markdown(
            "# 🔢 Object Counter\n"
            "Count objects in a photo — runs **100% locally** with free open-weight "
            "models (Grounding DINO, CLIP, SAM). No cloud, no paid APIs.\n\n"
            "**Mode A — by name:** type an object name (e.g. `pencil`, `coin`) → Detect & Count.\n\n"
            "**Mode B — by example:** Detect & Count first, then **click on one numbered "
            "box** in the result image — visually similar objects are counted via CLIP."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Input image", height=420)
                prompt = gr.Textbox(label="Object name / prompt (e.g. pencil)", value="pencil")
                box_thr = gr.Slider(0.05, 0.9, value=0.30, step=0.05,
                                    label="Detection confidence threshold (lower = more detections)")
                sim_thr = gr.Slider(0.5, 0.95, value=0.75, step=0.05,
                                    label="CLIP similarity threshold (Mode B: lower = more matches)")
                show_masks = gr.Checkbox(value=False, label="Show SAM segmentation masks")
                btn_detect_count = gr.Button("🔍 Detect & Count", variant="primary")
            with gr.Column():
                image_out = gr.Image(type="pil",
                                    label="Annotated result — click ON any object to count similar ones (Mode B)",
                                    height=420, interactive=False)
                count_out = gr.Markdown("Count will appear here.")
                table_out = gr.Textbox(label="Detections", lines=10, interactive=False)

        btn_detect_count.click(
            count_by_text,
            inputs=[image_in, prompt, box_thr, show_masks],
            outputs=[image_out, count_out, table_out])
        image_out.select(
            count_by_click,
            inputs=[prompt, box_thr, sim_thr, show_masks],
            outputs=[image_out, count_out, table_out])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_error=True)

