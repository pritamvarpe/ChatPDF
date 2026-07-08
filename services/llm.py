import os

import ollama
from ollama import ResponseError


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")


class OllamaServiceError(RuntimeError):
    pass


def build_prompt(question: str, context: str) -> str:
    return f"""Answer the question using only the context below.
If the context does not contain the answer, say that the PDF does not provide enough information.

Context:
{context}

Question:
{question}
"""


def generate_answer(question: str, context: str, model: str = DEFAULT_MODEL) -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": build_prompt(question, context)}],
        )
        return response["message"]["content"]
    except ResponseError as exc:
        if exc.status_code == 404:
            raise OllamaServiceError(
                f"Ollama model '{model}' is not installed. Run: ollama pull {model}"
            ) from exc
        raise OllamaServiceError(f"Ollama returned an error: {exc}") from exc
    except Exception as exc:
        raise OllamaServiceError(
            "Could not connect to Ollama. Make sure Ollama is installed and running."
        ) from exc
