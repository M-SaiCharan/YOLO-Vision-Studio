import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from comparison import annotate, compare, overlap
from comparison_service import run_batch


def detection(box=(0, 0, 12, 12), label="chair"):
    return {"label": label, "box": box, "confidence": 0.9}


class MatchingTests(unittest.TestCase):
    def test_duplicate_boxes_match_once_and_use_stable_index(self):
        d = detection()
        row = compare([d, d], [d], 1)
        self.assertEqual(row["pairs"], [{"a": 0, "b": 0, "iou": 1}])
        self.assertEqual(row["only_a"], [1])
        self.assertEqual(row["by_class"]["chair"], dict(matched=1, only_a=1, only_b=0))

    def test_best_overlap_first_and_class_boundary(self):
        d = detection()
        row = compare([detection((1, 1, 12, 12)), d], [d, detection(label="desk")], .5)
        self.assertEqual(row["pairs"][0]["a"], 1)
        self.assertEqual(row["only_b"], [1])

    def test_empty_disjoint_and_degenerate(self):
        self.assertEqual(compare([], [], .5)["by_class"], {})
        self.assertEqual(overlap((0, 0, 0, 0), (0, 0, 0, 0)), 0)
        self.assertFalse(compare([detection()], [detection((15, 15, 20, 20))], .5)["pairs"])

    def test_filters_hide_opposite_side(self):
        image = Image.new("RGB", (50, 50))
        row = compare([detection()], [], .5)
        self.assertEqual(annotate(image, row, "a", "Only B").tobytes(), image.tobytes())
        self.assertNotEqual(annotate(image, row, "a", "Only A").tobytes(), image.tobytes())


class BatchTests(unittest.TestCase):
    @patch("comparison_service.load_fresh_model")
    def test_warmup_batch_and_partial_decode_failure(self, load):
        import torch
        boxes = SimpleNamespace(xyxy=torch.empty(0, 4), cls=torch.empty(0), conf=torch.empty(0))
        prediction = SimpleNamespace(names={}, boxes=boxes, speed={"inference": 2.3})
        load.return_value.predict.return_value = [prediction]
        buffer = io.BytesIO()
        Image.new("RGB", (40, 40)).save(buffer, "PNG")
        data = buffer.getvalue()
        inputs = [("same.png", data), ("bad.jpg", b"invalid"), ("same.png", data)]
        report = run_batch(inputs, ["a.pt", "b.pt"], .4, .5)
        self.assertEqual(load.return_value.predict.call_count, 6)
        self.assertEqual([r["index"] for r in report["images"]], [0, 2])
        self.assertEqual(report["images"][1]["inference_ms"], {"a": 2.3, "b": 2.3})
        self.assertEqual(report["errors"][0]["filename"], "bad.jpg")
        self.assertEqual(json.loads(json.dumps(report)), report)
