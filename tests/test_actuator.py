import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rsg_mem.actuator import ImageOperationError, MultimodalActuator
from rsg_mem.models import ActionMode, TestCase


class ActuatorTests(unittest.TestCase):
    def test_registered_operations_create_provenanced_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (40, 30), "blue").save(source)
            test_case = TestCase(
                prompt="controlled",
                image_path=str(source),
                strategy_id="a",
                component_strategy_ids=("a",),
                action_mode=ActionMode.REUSE,
                metadata={
                    "image_operations": [
                        {"op": "resize", "width": 32, "height": 24},
                        {"op": "overlay_text", "text": "TOKEN", "x": 2, "y": 2},
                    ]
                },
            )
            result = MultimodalActuator(root / "artifacts").materialize(test_case)
            self.assertTrue(Path(result.image_path).is_file())
            with Image.open(result.image_path) as transformed:
                self.assertEqual(transformed.size, (32, 24))
            self.assertEqual(len(result.metadata["artifact_sha256"]), 64)

    def test_unknown_operation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (20, 20), "white").save(source)
            case = TestCase(
                prompt="controlled",
                image_path=str(source),
                strategy_id="a",
                component_strategy_ids=("a",),
                action_mode=ActionMode.REUSE,
                metadata={"image_operations": [{"op": "shell"}]},
            )
            with self.assertRaises(ImageOperationError):
                MultimodalActuator(root / "artifacts").materialize(case)


if __name__ == "__main__":
    unittest.main()
