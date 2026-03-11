from __future__ import annotations

from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore[import-untyped]

_TOKENIZER_ID = "Qwen/Qwen3.5-35B-A3B"
_tokenizer: PreTrainedTokenizerBase | None = None


def _get_tokenizer() -> PreTrainedTokenizerBase:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(_TOKENIZER_ID),
        )

    return _tokenizer


def render_messages(
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    tokenizer = _get_tokenizer()
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    return str(rendered)


def count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    token_ids: list[int] = tokenizer.encode(text)
    return len(token_ids)
