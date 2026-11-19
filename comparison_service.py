"""Batch detector comparison and Streamlit inspection workspace."""

import hashlib
import io
import json
from importlib.metadata import version

import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch

import config
from comparison import annotate, compare
from model_loader import load_fresh_model


def decode(data):
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def run_batch(inputs, names, confidence, match_iou, progress=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    settings = dict(device=device, conf=confidence, imgsz=640, iou=0.7,
                    max_det=300, rect=False, half=False, verbose=False)
    report = {
        "schema_version": 1, "models": dict(zip(("a", "b"), names)),
        "prediction_settings": settings, "match_iou": match_iou,
        "matching": "Greedy descending same-class IoU; ties use detection indices",
        "timing": "Ultralytics inference milliseconds, one warm-up per model",
        "hardware": torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU",
        "versions": {pkg: version(pkg) for pkg in ("ultralytics", "torch", "torchvision")},
        "images": [], "errors": [],
    }
    valid = []
    for index, (name, data) in enumerate(inputs):
        try:
            image = decode(data)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            report["errors"].append({"filename": name, "index": index, "error": str(exc)})
            continue
        valid.append((index, data))
        report["images"].append({"filename": name, "index": index,
                                 "sha256": hashlib.sha256(data).hexdigest(),
                                 "size": list(image.size), "inference_ms": {}})
    if not valid:
        return report

    for side_index, (side, name) in enumerate(zip(("a", "b"), names)):
        model = load_fresh_model(config.TASK_DETECT, model_name=name)
        model.predict(decode(valid[0][1]), **settings)
        for n, ((_, data), row) in enumerate(zip(valid, report["images"])):
            prediction = model.predict(decode(data), **settings)[0]
            row[side] = [{"box": box, "label": prediction.names[int(label)], "confidence": score}
                         for box, label, score in zip(prediction.boxes.xyxy.tolist(),
                                                      prediction.boxes.cls.tolist(),
                                                      prediction.boxes.conf.tolist())]
            row["inference_ms"][side] = prediction.speed["inference"]
            if progress:
                progress((side_index * len(valid) + n + 1) / (2 * len(valid)))
        del model
    for row in report["images"]:
        row.update(compare(row["a"], row["b"], match_iou))
    return report


def render():
    st.title("Compare models")
    st.write("Run two detectors on the same images and inspect where their boxes agree.")
    catalog = config.DETECTION_MODELS
    left, right = st.columns(2)
    a = left.selectbox("Detector A", list(catalog))
    b = right.selectbox("Detector B", [key for key in catalog if key != a])
    confidence = left.slider("Minimum confidence", 0.1, 1.0, 0.4, 0.05)
    threshold = right.slider("Minimum overlap for a match (IoU)", 0.05, 1.0, 0.5, 0.05,
                             help="Boxes must have the same class. IoU measures their intersection divided by their union.")
    files = st.file_uploader("Images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                             accept_multiple_files=True)
    inputs = [(f.name, f.getvalue()) for f in files]
    if not inputs:
        inputs = [(config.DEFAULT_IMAGE.name, config.DEFAULT_IMAGE.read_bytes())]
        st.caption("Using the bundled office photo. Upload images to compare your own batch.")
    signature = (a, b, confidence, threshold,
                 [(name, hashlib.sha256(data).digest()) for name, data in inputs])
    if st.button("Run comparison", type="primary", icon=":material/compare:"):
        bar = st.progress(0, text="Loading detectors and comparing images…")
        try:
            report = run_batch(inputs, [catalog[a], catalog[b]], confidence, threshold, bar.progress)
            st.session_state.comparison_run = (signature, report)
        except Exception as exc:
            st.session_state.pop("comparison_run", None)
            st.error(f"Comparison failed. Check model availability and device memory. {exc}")
        finally:
            bar.empty()
    saved = st.session_state.get("comparison_run")
    if saved and saved[0] == signature:
        show_results(saved[1], inputs)
    elif not files:
        preview = config.ROOT / "examples" / "office-comparison.jpg"
        if preview.exists():
            st.image(str(preview), caption="Recorded example · Unmatched office detections at 40% confidence and 0.50 match IoU.", width="stretch")


def show_results(report, inputs):
    for error in report["errors"]:
        st.warning(f"Skipped {error['filename']} — {error['error']}")
    rows = report["images"]
    st.download_button("Download JSON report", json.dumps(report, indent=2),
                       "detector-comparison.json", "application/json", icon=":material/download:")
    if not rows:
        return
    st.subheader("Batch results")
    for col, (label, key) in zip(st.columns(3), [
        ("Matched pairs", "pairs"), ("Only detector A", "only_a"), ("Only detector B", "only_b")
    ]):
        col.metric(label, sum(len(row[key]) for row in rows), border=True)
    summary = [{"Image": row["filename"], "Matched": len(row["pairs"]),
                "Only A": len(row["only_a"]), "Only B": len(row["only_b"]),
                "A (ms)": round(row["inference_ms"]["a"], 1),
                "B (ms)": round(row["inference_ms"]["b"], 1)} for row in rows]
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption(f"{report['hardware']} · Model inference time after warm-up. One measurement per image.")
    index = st.selectbox("Inspect image", range(len(rows)),
                         format_func=lambda i: f"{i + 1}. {rows[i]['filename']}")
    row = rows[index]
    image = decode(inputs[row["index"]][1])
    view = st.segmented_control("Show detections", ["All", "Matched", "Only A", "Only B"],
                                default="All") or "All"
    for col, side in zip(st.columns(2), ("a", "b")):
        with col.container(border=True):
            st.markdown(f"**{side.upper()}** · `{report['models'][side]}`")
            st.image(annotate(image, row, side, view), width="stretch")
    st.caption("Mint = matched pair · Amber = only A · Violet = only B. Agreement measures overlap, not accuracy.")
    if row["by_class"]:
        table = pd.DataFrame.from_dict(row["by_class"], orient="index")
        st.dataframe(table.rename_axis("Class").rename(columns={"matched": "Matched", "only_a": "Only A",
                                                                 "only_b": "Only B"}), width="stretch")
    else:
        st.info("Neither detector returned a box at this confidence level.")
