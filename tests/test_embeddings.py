import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rsg_mem.embeddings import HashingEmbedder, cosine, mean_vectors


class EmbeddingTests(unittest.TestCase):
    def test_hashing_embedding_is_deterministic_and_normalized(self):
        embedder = HashingEmbedder(64)
        first = embedder.encode_text("visual benchmark token")
        second = embedder.encode_text("visual benchmark token")
        self.assertEqual(first, second)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0)
        self.assertAlmostEqual(cosine(first, second), 1.0)

    def test_image_digest_and_mean_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.png"
            Image.new("RGB", (16, 16), "blue").save(path)
            embedder = HashingEmbedder(64)
            image = embedder.encode_image(str(path), "blue card")
            text = embedder.encode_text("blue card")
            combined = mean_vectors((image, text))
            self.assertEqual(len(combined), 64)
            self.assertGreater(cosine(combined, image), 0.0)


if __name__ == "__main__":
    unittest.main()
