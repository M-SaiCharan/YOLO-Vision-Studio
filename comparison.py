"""One-to-one detector comparison in original-image pixel coordinates."""

from collections import Counter
from itertools import product

from PIL import ImageDraw, ImageFont


def overlap(left, right):
    x1, y1, x2, y2 = left
    u1, v1, u2, v2 = right
    intersection = max(0, min(x2, u2) - max(x1, u1)) * max(0, min(y2, v2) - max(y1, v1))
    area = max(0, x2 - x1) * max(0, y2 - y1) + max(0, u2 - u1) * max(0, v2 - v1)
    union = area - intersection
    return intersection / union if union else 0.0


def compare(left, right, threshold):
    """Greedily pair the highest-IoU candidates first, breaking ties by box index."""
    candidates = []
    for (i, a), (j, b) in product(enumerate(left), enumerate(right)):
        if a["label"] == b["label"]:
            score = overlap(a["box"], b["box"])
            if score >= threshold:
                candidates.append((-score, i, j))

    used_a, used_b = set(), set()
    pairs = []
    counts = {label: Counter(matched=0, only_a=0, only_b=0)
              for label in sorted({d["label"] for d in left + right})}
    for negative_score, i, j in sorted(candidates):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append({"a": i, "b": j, "iou": -negative_score})
        counts[left[i]["label"]]["matched"] += 1

    only_a = [i for i in range(len(left)) if i not in used_a]
    only_b = [j for j in range(len(right)) if j not in used_b]
    for i in only_a:
        counts[left[i]["label"]]["only_a"] += 1
    for j in only_b:
        counts[right[j]["label"]]["only_b"] += 1
    return {"a": left, "b": right, "pairs": pairs, "only_a": only_a,
            "only_b": only_b, "by_class": counts}


def annotate(image, result, side, view="All"):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, round(canvas.width / 480))
    font = ImageFont.load_default(size=max(12, stroke * 5))
    unmatched = set(result[f"only_{side}"])
    for index, det in enumerate(result[side]):
        unique = index in unmatched
        category = f"Only {side.upper()}" if unique else "Matched"
        if view not in ("All", category):
            continue
        color = ("#ffb35c" if side == "a" else "#c5a2ff") if unique else "#59dbc0"
        box = det["box"]
        draw.rectangle(box, outline=color, width=stroke)
        label = f"{det['label']} {det['confidence']:.0%}"
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2] + 6, bounds[3] + 4
        x = max(0, min(box[0], canvas.width - width))
        y = max(0, box[1] - height)
        draw.rectangle((x, y, x + width, y + height), fill=color)
        draw.text((x + 3, y + 1), label, font=font, fill="#101820")
    return canvas
