"""Python port of the dataviz skill's validate_palette.js checks (lightness band, chroma floor, CVD and normal-vision
separation in OKLab x100 with Machado 2009 matrices, contrast vs surface)."""
import itertools, math, sys
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
M = {"protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
     "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
     "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]}
def srgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
def s2l(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
def lin(h): return [s2l(c) for c in srgb(h)]
def oklab_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]
def sim(h, k):
    r, g, b = lin(h); A = M[k]
    return [min(1, max(0, A[i][0] * r + A[i][1] * g + A[i][2] * b)) for i in range(3)]
def dE(a, b, k=None):
    x = oklab_lin(sim(a, k) if k else lin(a)); y = oklab_lin(sim(b, k) if k else lin(b))
    return 100 * math.dist(x, y)
def lum(h): r, g, b = lin(h); return 0.2126 * r + 0.7152 * g + 0.0722 * b
def contrast(a, b): hi, lo = sorted([lum(a), lum(b)], reverse=True); return (hi + 0.05) / (lo + 0.05)
def check(pal, mode, surface, pairs="all"):
    lo, hi = BAND[mode]
    out = {}
    L = {c: oklab_lin(lin(c))[0] for c in pal}; C = {c: math.hypot(*oklab_lin(lin(c))[1:]) for c in pal}
    out["band"] = [(c, round(L[c], 3)) for c in pal if not lo <= L[c] <= hi]
    out["chroma<0.10"] = [(c, round(C[c], 3)) for c in pal if C[c] < 0.10]
    pl = list(itertools.combinations(range(len(pal)), 2)) if pairs == "all" else [(i, i + 1) for i in range(len(pal) - 1)]
    out["cvd_worst"] = min((round(dE(pal[i], pal[j], k), 1), k, pal[i], pal[j]) for k in ("protan", "deutan") for i, j in pl)
    out["tritan_worst"] = min(round(dE(pal[i], pal[j], "tritan"), 1) for i, j in pl)
    out["normal_worst"] = min((round(dE(pal[i], pal[j]), 1), pal[i], pal[j]) for i, j in pl)
    out["contrast<3"] = [(c, round(contrast(c, surface), 2)) for c in pal if contrast(c, surface) < 3]
    return out
if __name__ == "__main__":
    pal = sys.argv[1].split(","); mode = sys.argv[2]; surface = sys.argv[3]; pairs = sys.argv[4] if len(sys.argv) > 4 else "all"
    for k, v in check(pal, mode, surface, pairs).items(): print(f"  {k}: {v}")
