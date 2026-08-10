"""
Unit tests for agents.explanation.llm_client module
File: tests/test_llm.py
"""

import unittest
from unittest.mock import MagicMock, patch
from agents.explanation.llm_client import GeminiClient, GeminiAPIError

class TestGeminiClient(unittest.TestCase):

    def test_missing_api_key_raises_error(self):
        client = GeminiClient(api_key="")
        with self.assertRaises(GeminiAPIError):
            client.generate("Test prompt")

    @patch("agents.explanation.llm_client.genai.Client")
    def test_successful_generation(self, mock_genai_client):
        # Mock successful API response
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a test explanation."
        mock_instance.models.generate_content.return_value = mock_response
        mock_genai_client.return_value = mock_instance

        client = GeminiClient(api_key="mock_key_for_testing")
        client.client = mock_instance
        client.sdk_type = "new"

        res = client.generate("Explain HbA1c 8.2%")
        self.assertEqual(res, "This is a test explanation.")

    @patch("agents.explanation.llm_client.time.sleep", return_value=None)
    def test_retry_on_failure(self, mock_sleep):
        mock_instance = MagicMock()
        mock_instance.models.generate_content.side_effect = Exception("Transient Network Error")

        client = GeminiClient(api_key="mock_key_for_testing")
        client.client = mock_instance
        client.sdk_type = "new"

        with self.assertRaises(GeminiAPIError):
            client.generate("Explain HbA1c 8.2%", max_retries=3)

        self.assertGreaterEqual(mock_instance.models.generate_content.call_count, 1)

if __name__ == "__main__":
    unittest.main()
