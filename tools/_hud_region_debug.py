"""
Crops each HUD region from a real gameplay frame and saves the crops.
Run this, view the output images, then update region fractions accordingly.
"""
from pathlib import Path
import cv2
import numpy as np

FRAME  = Path("data/screenshots/hud_frames/frame_50pct_raw.png")
OUTDIR = Path("data/screenshots/hud_frames/crops")
OUTDIR.mkdir(exist_ok=True)

REGIONS = {
    "wave":        (0.22, 0.43, 0.60, 0.49),
    "player_hp":   (0.02, 0.43, 0.28, 0.50),
    "opp_hp":      (0.62, 0.43, 0.97, 0.50),
    "player_mana": (0.00, 0.49, 0.07, 0.72),
}

frame = cv2.imread(str(FRAME))
h, w  = frame.shape[:2]
print(f"Frame: {w}x{h}")

for name, (l, t, r, b) in REGIONS.items():
    x1, y1 = int(l * w), int(t * h)
    x2, y2 = int(r * w), int(b * h)
    crop = frame[y1:y2, x1:x2]
    # Upscale 4x so small regions are readable
    big  = cv2.resize(crop, (crop.shape[1]*4, crop.shape[0]*4),
                      interpolation=cv2.INTER_NEAREST)
    out  = OUTDIR / f"{name}.png"
    cv2.imwrite(str(out), big)
    print(f"  {name:15s} px: ({x1},{y1})-({x2},{y2})  size: {x2-x1}x{y2-y1}")

# Also save a full annotated frame with pixel rulers
annotated = frame.copy()
COLORS = {"wave":(0,200,0), "player_hp":(0,0,255),
          "opp_hp":(255,50,50), "player_mana":(0,200,255)}
for name, (l, t, r, b) in REGIONS.items():
    x1,y1 = int(l*w), int(t*h)
    x2,y2 = int(r*w), int(b*h)
    cv2.rectangle(annotated,(x1,y1),(x2,y2),COLORS[name],2)
    cv2.putText(annotated,f"{name} y={y1}-{y2}",(x1,max(y1-3,10)),
                cv2.FONT_HERSHEY_SIMPLEX,0.3,COLORS[name],1)

# Draw horizontal guide lines every 10% of height
for pct in range(0,11):
    y = int(pct/10 * h)
    cv2.line(annotated,(0,y),(w,y),(200,200,200),1)
    cv2.putText(annotated,f"{pct*10}%",(2,y-2),
                cv2.FONT_HERSHEY_SIMPLEX,0.25,(200,200,200),1)

cv2.imwrite(str(OUTDIR / "annotated_ruler.png"), annotated)
print(f"\nCrops saved to {OUTDIR}")
print("Check annotated_ruler.png to see % lines vs. region boxes.")