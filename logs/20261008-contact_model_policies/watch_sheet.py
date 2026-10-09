"""One contact sheet per run from its watch strips: the median rollout's row (or the worst one's) of every watched
checkpoint, halved, stacked in iteration order.
    python3 logs/20261008-contact_model_policies/watch_sheet.py <tag> [median|worst] [out.jpg]"""
import glob, os, sys
from PIL import Image
tag, row = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "median")
D = f"/home/humanoid/Programs/hand/docs/experiments/20261008-contact_model_policies/watch/{tag}"
fs = sorted(glob.glob(f"{D}/it*_strip.jpg"))
ims = []
for f in fs:
    im = Image.open(f)
    w, h = im.size
    im = im.crop((0, 0, w, h // 2) if row == "median" else (0, h // 2, w, h))
    ims.append(im.resize((w // 2, h // 4)))
sheet = Image.new("RGB", (ims[0].size[0], sum(i.size[1] for i in ims)), "white")
y = 0
for im in ims:
    sheet.paste(im, (0, y)); y += im.size[1]
out = sys.argv[3] if len(sys.argv) > 3 else f"{D}/sheet_{row}.jpg"
sheet.save(out, quality=85)
print(out, sheet.size)
