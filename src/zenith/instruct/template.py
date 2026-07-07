"""The chat/instruction prompt template.

A single, explicit template shared by fine-tuning and inference — so the string the
model is trained on is exactly the string it sees at chat time. Plain text (no
special vocabulary needed), so it works with the byte-level tokenizer unchanged; the
end of a response is marked by the tokenizer's EOS id.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ChatTemplate"]


@dataclass(frozen=True)
class ChatTemplate:
    """Formats instructions the way the model is fine-tuned to expect them."""

    user_header: str = "### Instruction:\n"
    assistant_header: str = "\n\n### Response:\n"

    def format_prompt(self, instruction: str) -> str:
        """The prompt half — everything up to where the response begins."""
        return f"{self.user_header}{instruction}{self.assistant_header}"

    def format_example(self, instruction: str, response: str) -> str:
        """A full training example: prompt + response (EOS is added by the dataset)."""
        return f"{self.format_prompt(instruction)}{response}"
