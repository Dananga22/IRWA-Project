"""
MedExplain AI — LLM Explanation Client (Gemini API Integration)
Module: agents.explanation.llm_client
"""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Configure logging according to AGENTS.md rules
logger = logging.getLogger("MedExplain.LLMClient")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# SDK Support check
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False


class GeminiAPIError(Exception):
    """Custom exception raised when the Gemini API fails after max retries."""
    pass


class GeminiClient:
    """Client wrapper for Google Gemini API with logging, retries, and error handling."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY is not configured in environment variables.")

        if HAS_GENAI_SDK and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def generate(self, prompt: str, temperature: float = 0.2, max_retries: int = 3) -> str:
        """Generate text using Gemini API with exponential backoff retries and logging.

        Args:
            prompt: Text prompt to send to the LLM.
            temperature: Sampling temperature (default 0.2 for deterministic output).
            max_retries: Maximum retry attempts on transient failure.

        Returns:
            Generated response text string.

        Raises:
            GeminiAPIError: If API key is missing or calls fail after retries.
        """
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise GeminiAPIError("GEMINI_API_KEY is missing or invalid. Please set your key in .env.")

        if not self.client:
            raise GeminiAPIError("Google GenAI SDK (google-genai) is not installed.")

        prompt_len = len(prompt)
        last_exception: Optional[Exception] = None

        candidate_models = [self.model_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        candidate_models = list(dict.fromkeys(candidate_models))

        for model in candidate_models:
            for attempt in range(1, max_retries + 1):
                start_time = time.time()
                try:
                    logger.info(f"Sending prompt to Gemini model '{model}' (Attempt {attempt}/{max_retries}, Prompt len: {prompt_len} chars)")

                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=temperature)
                    )
                    text_result = response.text

                    if not text_result:
                        raise GeminiAPIError("Gemini API returned an empty response string.")

                    latency = time.time() - start_time
                    resp_len = len(text_result)
                    logger.info(f"Gemini API Call Successful! Model: {model} | Latency: {latency:.2f}s | Response len: {resp_len} chars")
                    return text_result

                except Exception as e:
                    latency = time.time() - start_time
                    last_exception = e
                    err_str = str(e)
                    logger.warning(f"Model {model} Attempt {attempt}/{max_retries} failed after {latency:.2f}s: {e}")

                    if "404" in err_str or "NOT_FOUND" in err_str or "API_KEY_INVALID" in err_str or "400" in err_str:
                        logger.warning(f"Model {model} unavailable or API key error. Trying next candidate model...")
                        break

                    if attempt < max_retries:
                        sleep_time = 2 ** (attempt - 1)
                        logger.info(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)

        raise GeminiAPIError(f"Gemini API call failed for all models. Last Error: {last_exception}")
