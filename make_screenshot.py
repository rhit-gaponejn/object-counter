"""Generate a portfolio screenshot: synthetic scattered pencils, detected & counted."""
from PIL import Image, ImageDraw
import random
import app
import detector

random.seed(42)

W, H = 1000, 620
img = Image.new("RGB", (W, H), (188, 182, 172))  # wooden-floor-ish grey
d = ImageDraw.Draw(img)
# subtle floor grain
for y in range(0, H, 26):
    d.line([(0, y + random.randint(-4, 4)), (W, y + random.randint(-4, 4))],
           fill=(178, 172, 162), width=3)

COLORS = [(200, 55, 55), (66, 110, 220), (230, 170, 40), (80, 160, 70),
          (160, 90, 190), (40, 180, 190), (220, 120, 60), (100, 100, 110)]
PENCIL_LEN, PENCIL_W = 300, 26

def draw_pencil(cx, cy, angle, color):
    """Draw a pencil (hex body + wood tip + graphite + eraser band)."""
    import math
    rad = math.radians(angle)
    dx, dy = math.cos(rad), math.sin(rad)
    px, py = -dy, dx  # perpendicular
    def pt(t, s):
        return (cx + dx * t + px * s, cy + dy * t + py * s)
    # body (two tones for hex facets)
    d.polygon([pt(-PENCIL_LEN/2, -PENCIL_W/2), pt(PENCIL_LEN/2 - 40, -PENCIL_W/2),
               pt(PENCIL_LEN/2 - 40, PENCIL_W/2), pt(-PENCIL_LEN/2, PENCIL_W/2)],
              fill=color)
    d.polygon([pt(-PENCIL_LEN/2, -PENCIL_W/6), pt(PENCIL_LEN/2 - 40, -PENCIL_W/6),
               pt(PENCIL_LEN/2 - 40, PENCIL_W/6), pt(-PENCIL_LEN/2, PENCIL_W/6)],
              fill=tuple(min(255, c + 30) for c in color))
    # wooden tip
    d.polygon([pt(PENCIL_LEN/2 - 40, -PENCIL_W/2), pt(PENCIL_LEN/2, 0),
               pt(PENCIL_LEN/2 - 40, PENCIL_W/2)], fill=(222, 190, 140))
    # graphite point
    d.polygon([pt(PENCIL_LEN/2 - 10, -6), pt(PENCIL_LEN/2, 0), pt(PENCIL_LEN/2 - 10, 6)],
              fill=(60, 60, 60))
    # eraser
    d.polygon([pt(-PENCIL_LEN/2, -PENCIL_W/2), pt(-PENCIL_LEN/2 + 26, -PENCIL_W/2),
               pt(-PENCIL_LEN/2 + 26, PENCIL_W/2), pt(-PENCIL_LEN/2, PENCIL_W/2)],
              fill=(240, 150, 160))
    d.polygon([pt(-PENCIL_LEN/2 + 26, -PENCIL_W/2), pt(-PENCIL_LEN/2 + 36, -PENCIL_W/2),
               pt(-PENCIL_LEN/2 + 36, PENCIL_W/2), pt(-PENCIL_LEN/2 + 26, PENCIL_W/2)],
              fill=(190, 190, 190))

spots = [(170, 140, -20), (500, 90, 15), (830, 150, 160), (260, 330, 35),
         (620, 320, -50), (900, 360, 25), (120, 480, 70), (450, 500, 5),
         (760, 500, -80), (330, 200, 100)]
for cx, cy, angle in spots:
    draw_pencil(cx, cy, angle, COLORS[spots.index((cx, cy, angle)) % len(COLORS)])

annotated, count_md, table = app.count_by_text(img, "pencil", 0.28, False)
print(count_md)
annotated.save("screenshot_object_counter.jpg", quality=92)
print("saved screenshot_object_counter.jpg", annotated.size)