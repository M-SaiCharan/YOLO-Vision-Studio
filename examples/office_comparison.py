"""Regenerate the README figure and its detection report from the repo root."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config
from comparison import annotate
from comparison_service import decode, run_batch


def main():
    output = Path(__file__).resolve().parent
    data = config.DEFAULT_IMAGE.read_bytes()
    report = run_batch([(config.DEFAULT_IMAGE.name, data)], ["yolo26n.pt", "rtdetr-l.pt"], .4, .5)
    row = report["images"][0]
    image = decode(data)
    panel_width = 760
    panel_height = round(image.height * panel_width / image.width)
    canvas = Image.new("RGB", (panel_width * 2 + 48, panel_height + 94), "#111b25")
    draw = ImageDraw.Draw(canvas)
    for x, side, color in [(16, "a", "#ffb35c"), (32 + panel_width, "b", "#c5a2ff")]:
        panel = annotate(image, row, side, f"Only {side.upper()}")
        canvas.paste(panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS), (x, 52))
        draw.text((x, 12), f"{side.upper()}   {report['models'][side]}", fill=color,
                  font=ImageFont.load_default(size=23))
        count = len(row[f"only_{side}"])
        draw.text((x, panel_height + 59), f"{count} unmatched detections", fill="#edf4f5",
                  font=ImageFont.load_default(size=18))
    canvas.save(output / "office-comparison.jpg", quality=90)
    (output / "office-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
