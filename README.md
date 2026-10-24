# 🔢 Object Counter

Count objects in a photo — **100% local, free, no paid APIs**. Point it at a picture
of spilled pencils, a jar of coins, or a parking lot, and get an object count with
annotated, numbered detections.

Powered by open-weight models running on your own hardware:

| Model | Role |
|---|---|
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | Open-vocabulary detection — find any object by name, no training required |
| [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) | Count objects visually similar to a clicked example |
| [SAM](https://huggingface.co/facebook/sam-vit-base) | Optional pretty segmentation masks |

## Quick start

Requirements: Python 3.10–3.13, an NVIDIA GPU (4 GB+ VRAM) recommended (CPU works
but is slower).

```bash
git clone <this repo>
cd object-counter
python -m venv .venv
.venv\Scripts\activate          # Windows (use `source .venv/bin/activate` on Linux/macOS)
pip install -r requirements.txt
python app.py
```

On Windows you can also just **double-click `run_app.bat`**.

Then open **http://127.0.0.1:7860** in your browser.

> The first run downloads model weights (~1.5 GB total) to your Hugging Face cache.
> Nothing is sent to any server afterwards — inference is entirely local.

## How to use

- **Mode A — count by name:** load a photo, type the object name (`pencil`, `coin`,
  `bottle`, `person`...), click **Detect & Count**. You get the count, numbered boxes
  on the annotated image, and a detections table.
- **Mode B — count by example:** after detecting, **click on one of the objects** in
  the result image. Every detection visually similar to your clicked example (via
  CLIP embeddings) is counted. Great for filtering false positives or separating
  similar-looking objects.

### Tips for accuracy

- Lower the **detection confidence threshold** (e.g. 0.20) for small or cluttered objects.
- Raise the **CLIP similarity threshold** in Mode B to narrow the match set.
- Overlapping boxes are deduplicated with IoU + containment NMS, so dense piles
  don't inflate the count.
- Dense, heavily overlapping piles are the hardest case for any open-vocabulary
  detector — Mode B is your friend there.

## How it works

1. Grounding DINO takes a text prompt and finds every matching region in the image,
   producing boxes + confidence scores (no training data needed).
2. A custom NMS pass (IoU ≥ 0.5 or ≥ 80% containment) removes duplicate boxes.
3. In Mode B, each detection crop is embedded with CLIP; crops whose embedding is
   within the similarity threshold of the clicked example are kept.
4. Optionally, SAM segments each box for translucent colored masks.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Gradio UI (localhost) + annotation drawing |
| `detector.py` | Grounding DINO text-prompt detection + NMS |
| `similarity.py` | CLIP matching for the click-an-example flow |
| `segmenter.py` | SAM segmentation masks |
| `smoke_test.py` | End-to-end test with a synthetic image |
| `run_app.bat` | One-click Windows launcher |

## Testing

```bash
python smoke_test.py
```

Generates a synthetic image with 6 coins and verifies detection, NMS dedup,
CLIP matching, SAM segmentation, and click-point mapping.

## License

[MIT](LICENSE) — the underlying models have their own licenses
(Grounding DINO: Apache 2.0, CLIP: MIT, SAM: Apache 2.0) via Hugging Face.

