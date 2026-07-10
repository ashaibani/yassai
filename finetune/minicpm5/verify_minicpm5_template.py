"""Modal probe for MiniCPM5 tokenizer/chat-template behavior.

This intentionally uses the same dependency family as the training image.
"""

from __future__ import annotations

import json

import modal


app = modal.App("yassai-minicpm5-template-probe")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.57.3",
        "accelerate==1.11.0",
        "sentencepiece==0.2.1",
        "protobuf==6.33.0",
    )
)


@app.function(image=image, timeout=20 * 60)
def probe() -> dict:
    from transformers import AutoTokenizer

    model_id = "openbmb/MiniCPM5-1B"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    messages = [
        {"role": "system", "content": "Use run_python for arithmetic."},
        {"role": "user", "content": "What is 17*23?"},
    ]
    with_assistant = messages + [
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"run_python","arguments":{"code":"print(17*23)"}}</tool_call>',
        }
    ]
    result = {
        "transformers_tokenizer_class": type(tok).__name__,
        "chat_template_present": tok.chat_template is not None,
        "chat_template": tok.chat_template,
        "special_ids": {
            "bos": tok.bos_token_id,
            "eos": tok.eos_token_id,
            "pad": tok.pad_token_id,
            "im_start": tok.convert_tokens_to_ids("<|im_start|>"),
            "im_end": tok.convert_tokens_to_ids("<|im_end|>"),
            "tool_call_xml_start": tok.convert_tokens_to_ids("<tool_call>"),
            "tool_call_xml_end": tok.convert_tokens_to_ids("</tool_call>"),
            "no_think": tok.convert_tokens_to_ids("/no_think"),
            "think": tok.convert_tokens_to_ids("/think"),
        },
    }
    for name, kwargs in {
        "no_think_prompt": {"enable_thinking": False, "add_generation_prompt": True},
        "think_prompt": {"enable_thinking": True, "add_generation_prompt": True},
        "assistant_train_render": {"enable_thinking": False, "add_generation_prompt": False},
    }.items():
        source = with_assistant if name == "assistant_train_render" else messages
        rendered = tok.apply_chat_template(source, tokenize=False, **kwargs)
        tokenized = tok.apply_chat_template(source, tokenize=True, return_tensors=None, **kwargs)
        result[name] = {
            "text": rendered,
            "token_ids_prefix": tokenized[:24],
            "token_ids_suffix": tokenized[-24:],
            "token_count": len(tokenized),
        }
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(probe.remote(), ensure_ascii=False, indent=2))
