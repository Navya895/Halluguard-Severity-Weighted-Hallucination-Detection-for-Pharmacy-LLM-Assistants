import unittest
from unittest.mock import Mock, patch

from severity_scorer_llm import compute_severity_llm, parse_severity, query_severity
from evaluate_severity import summarize


class SeverityTests(unittest.TestCase):
    def test_leading_label(self):
        self.assertEqual(parse_severity(" HIGH. Explanation"), "HIGH")
        self.assertEqual(parse_severity("low"), "LOW")
        self.assertEqual(parse_severity("Severity: HIGH; Consequence: temporary harm."), "HIGH")

    def test_invalid_outputs_do_not_become_medium(self):
        for text in ("", "<|im_end|>", "Maybe HIGH", "HIGHLIGHT"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_severity(text)

    @patch("severity_scorer_llm.requests.post")
    def test_transport_and_public_scorer(self, post):
        response = Mock()
        response.json.return_value = {"response": "MEDIUM"}
        post.return_value = response
        self.assertEqual(compute_severity_llm("A test claim"), "MEDIUM")
        payload = post.call_args.kwargs["json"]
        self.assertTrue(payload["raw"])
        self.assertTrue(payload["prompt"].startswith("[INST] "))
        self.assertTrue(payload["prompt"].endswith(" [/INST]"))
        self.assertIn("A test claim", payload["prompt"])
        response.raise_for_status.assert_called_once()

    def test_invalid_predictions_count_as_errors(self):
        result = summarize([
            {"human_label": "LOW", "prediction": "LOW"},
            {"human_label": "HIGH", "prediction": "INVALID"},
        ])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["invalid_predictions"], 1)
        self.assertEqual(result["per_class"]["HIGH"]["recall"], 0.0)

    @patch("severity_scorer_llm.requests.post")
    def test_comparison_model_uses_its_native_template(self, post):
        post.return_value.json.return_value = {"response": "Severity: HIGH; Consequence: harm."}
        self.assertEqual(compute_severity_llm("A test claim", model="qwen2.5:7b-instruct"), "HIGH")
        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["raw"])
        self.assertEqual(payload["model"], "qwen2.5:7b-instruct")
        self.assertNotIn("[INST]", payload["prompt"])


if __name__ == "__main__":
    unittest.main()
