import hashlib
import unittest

from xai_regression.provenance import artifact_fingerprint


class ProvenanceTests(unittest.TestCase):
    def test_fingerprint_identifies_exact_uploaded_bytes(self):
        payload = b"version-aware-model"
        result = artifact_fingerprint("model.joblib", payload)

        self.assertEqual(result["filename"], "model.joblib")
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["short_sha256"], result["sha256"][:12])
        self.assertEqual(result["bytes"], len(payload))


if __name__ == "__main__":
    unittest.main()
