"""
Local LLM client using HuggingFace Transformers (Qwen2.5-0.5B).

Note: We keep the class name `OllamaClient` so other modules do not need
to change, but this implementation no longer depends on Ollama. It loads
the model directly with `transformers` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any]


class OllamaClient:
    """Wrapper over a local Qwen model loaded via HuggingFace Transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    def close(self) -> None:
        # Kept for API compatibility; nothing to close for Transformers model.
        pass

    def generate(
        self,
        system_prompt: str,
        messages: List[LLMMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a response using the local Qwen model."""
        chat_messages: List[Dict[str, str]] = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            chat_messages.append({"role": m.role, "content": m.content})

        inputs = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens or 256,
            "do_sample": temperature > 0,
            "temperature": float(temperature),
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Take only the generated continuation.
        gen_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return LLMResponse(text=text, raw={"generated_ids": gen_ids.tolist()})


__all__ = ["OllamaClient", "LLMMessage", "LLMResponse"]


