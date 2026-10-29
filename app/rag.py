"""Retrieval + generation: answer questions and build practice quizzes with Claude."""

import json

import anthropic

from app.config import CLAUDE_MODEL
from app.vector_store import SearchResult, VectorStore

ANSWER_SYSTEM = """You are a study assistant for a university student.
Answer using ONLY the numbered course excerpts provided. Cite the excerpts you rely on
inline, like [1] or [2][3]. If the excerpts do not contain the answer, say so plainly
instead of guessing. Answer in the same language as the question (Hebrew or English).
Be clear and concise, the way a good teaching assistant would explain it."""

QUIZ_SYSTEM = """You write practice exam questions from course material.
Every question must be answerable from the excerpts alone. Write in the language of the
excerpts. Mix recall questions with questions that test understanding."""

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "correct_index": {"type": "integer"},
                    "explanation": {"type": "string"},
                    "page": {"type": "integer"},
                },
                "required": ["question", "options", "correct_index", "explanation", "page"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

# Server-side fallback: if a request is declined by a safety classifier, the API
# re-runs it on a suitable fallback model in the same call.
FALLBACK_ARGS = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


class RefusedError(RuntimeError):
    pass


def format_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{i}] ({r.document}, page {r.page})\n{r.text}" for i, r in enumerate(results, start=1)
    )


def _text(response) -> str:
    if response.stop_reason == "refusal":
        raise RefusedError("Claude declined this request.")
    return "".join(block.text for block in response.content if block.type == "text")


class StudyAssistant:
    def __init__(self, store: VectorStore, client: anthropic.Anthropic | None = None):
        self.store = store
        self.client = client or anthropic.Anthropic()

    def ask(self, question: str, document: str | None = None) -> dict:
        results = self.store.search(question, document=document)
        if not results:
            return {"answer": "No documents uploaded yet. Upload a PDF first.", "sources": []}

        response = self.client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=ANSWER_SYSTEM,
            output_config={"effort": "medium"},
            messages=[{
                "role": "user",
                "content": f"<excerpts>\n{format_context(results)}\n</excerpts>\n\nQuestion: {question}",
            }],
            **FALLBACK_ARGS,
        )
        return {
            "answer": _text(response),
            "sources": [
                {"id": i, "document": r.document, "page": r.page, "text": r.text, "score": round(r.score, 3)}
                for i, r in enumerate(results, start=1)
            ],
        }

    def quiz(self, document: str, num_questions: int = 5) -> list[dict]:
        excerpts = self.store.sample(document)
        if not excerpts:
            raise ValueError(f"Unknown document: {document}")

        response = self.client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=QUIZ_SYSTEM,
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": QUIZ_SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": (
                    f"<excerpts>\n{format_context(excerpts)}\n</excerpts>\n\n"
                    f"Write {num_questions} multiple-choice questions with 4 options each. "
                    "Set page to the page number the answer comes from."
                ),
            }],
            **FALLBACK_ARGS,
        )
        return json.loads(_text(response))["questions"]
