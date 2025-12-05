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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


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
        model_name: str = "google/gemma-3-1b-it",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Loading model on device: {self.device}")
        
        # Try to load config first to determine model type
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            # If config doesn't have model_type, set it based on the base model (gemma3)
            if not hasattr(config, 'model_type') or config.model_type is None:
                config.model_type = "gemma3"
        except:
            # If config loading fails, we'll try loading the model anyway
            config = None
        
        # Load model with device_map="auto" for GPU if available (handles device placement automatically)
        # Otherwise, load on CPU
        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,  # Some models require this
        }
        if config:
            load_kwargs["config"] = config
        
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = None
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception as e:
            # If loading with device_map fails, try without it
            print(f"Warning: Failed to load with device_map, trying without: {e}")
            load_kwargs.pop("device_map", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            self.model = self.model.to(self.device)
        
        # If device_map wasn't used, explicitly move to device
        if load_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
        
        print(f"Model loaded on device: {self.device}")

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
        """Generate a response using Gemma model.
        
        Note: Gemma doesn't support 'system' role. System instructions should be
        included in the first user message instead.
        """
        chat_messages: List[Dict[str, str]] = []
        
        # Gemma doesn't support 'system' role - include system prompt in first user message
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]
        
        # Build messages alternating user/assistant
        # If we have a system prompt, prepend it to the first user message
        if user_messages:
            first_user_content = user_messages[0].content
            if system_prompt:
                first_user_content = f"{system_prompt}\n\n{first_user_content}"
            chat_messages.append({"role": "user", "content": first_user_content})
            
            # Add remaining messages in order
            for i in range(1, len(user_messages)):
                if i - 1 < len(assistant_messages):
                    chat_messages.append({"role": "assistant", "content": assistant_messages[i - 1].content})
                chat_messages.append({"role": "user", "content": user_messages[i].content})
        else:
            # No user messages, create one with system prompt
            if system_prompt:
                chat_messages.append({"role": "user", "content": system_prompt})

        # Use return_dict=True to ensure we get a proper dictionary (official pattern)
        inputs = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Get end_of_turn token ID for Gemma models (token ID 106)
        # According to Gemma docs: "<end_of_turn>\n" is the turn separator
        end_of_turn_token_id = None
        try:
            end_of_turn_encoded = self.tokenizer.encode("<end_of_turn>", add_special_tokens=False)
            if end_of_turn_encoded:
                end_of_turn_token_id = end_of_turn_encoded[0]
        except:
            pass
        
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": min(max_tokens or 300, 300) if max_tokens is not None else 300,  # Cap at 300 tokens
            "do_sample": temperature > 0,
            "temperature": float(temperature),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.5,  # Prevent repetitive generation
            "no_repeat_ngram_size": 3,  # Prevent repeating 3-grams
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Take only the generated continuation (new tokens only) - official pattern
        input_length = inputs["input_ids"].shape[-1]
        gen_ids = outputs[0][input_length:]
        
        # Stop at end_of_turn token if present (Gemma's turn separator)
        # This is critical - Gemma uses <end_of_turn> to signal end of response
        if end_of_turn_token_id and end_of_turn_token_id in gen_ids.tolist():
            end_idx = gen_ids.tolist().index(end_of_turn_token_id)
            gen_ids = gen_ids[:end_idx]
        
        # Decode the generated text - return raw output without any post-processing
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        # Remove end_of_turn token from text if it appears (shouldn't if we stopped correctly)
        text = text.replace("<end_of_turn>", "").strip()

        return LLMResponse(text=text, raw={"generated_ids": gen_ids.tolist()})


__all__ = ["OllamaClient", "LLMMessage", "LLMResponse"]


