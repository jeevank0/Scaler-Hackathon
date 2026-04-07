from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"
load_dotenv(ENV_FILE)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Set it in {ENV_FILE}."
        )
    return value


def extract_output_text(response: object) -> str:
    text = (getattr(response, "output_text", "") or "").strip()
    if text:
        return text

    parts: list[str] = []
    output_items = getattr(response, "output", None)
    if output_items is None and hasattr(response, "model_dump"):
        payload = response.model_dump()
        output_items = payload.get("output", [])

    for item in output_items or []:
        if isinstance(item, dict):
            content_items = item.get("content", [])
        else:
            content_items = getattr(item, "content", []) or []

        for content in content_items:
            if isinstance(content, dict):
                chunk = content.get("text")
            else:
                chunk = getattr(content, "text", None)
            if chunk:
                parts.append(str(chunk))
    return "\n".join(parts).strip()


def get_response_error_message(response: object) -> str | None:
    error_obj = getattr(response, "error", None)
    if not error_obj:
        return None

    if isinstance(error_obj, dict):
        code = error_obj.get("code")
        message = error_obj.get("message")
    else:
        code = getattr(error_obj, "code", None)
        message = getattr(error_obj, "message", None)

    if code and message:
        return f"{code}: {message}"
    if message:
        return str(message)
    return str(error_obj)


def select_client_config() -> tuple[str, str]:
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if "huggingface.co" in base_url.lower():
        raise RuntimeError(
            "Hugging Face router is disabled for LLM calls in this script. "
            "Set API_BASE_URL to https://api.openai.com/v1"
        )

    if not openai_api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY for the configured API_BASE_URL."
        )
    return base_url, openai_api_key


def main() -> None:
    base_url, api_key = select_client_config()
    model = os.getenv("MODEL_NAME", "gpt-5-nano").strip()
    prompt = " ".join(sys.argv[1:]).strip() or "write a haiku about ai"

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.responses.create(
        model=model,
        input=prompt,
        store=True,
    )

    error_message = get_response_error_message(response)
    if error_message:
        raise RuntimeError(f"Responses API failed: {error_message}")

    output_text = extract_output_text(response)
    if not output_text:
        raise RuntimeError("OpenAI response did not contain output text.")

    print(output_text)


if __name__ == "__main__":
    main()
