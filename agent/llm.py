"""Factory per il LLM 'cervello' dell'agente: modello Qwen2.5-Instruct
eseguito localmente su GPU (nessuna chiamata a API a pagamento).

`langchain_huggingface.ChatHuggingFace` con backend "pipeline" (locale) non
inoltra lo schema dei tool al chat template - lo fa solo il backend
"endpoint" (Inference API a pagamento). Il chat template di Qwen2.5 gestisce
il tool-calling emettendo blocchi XML `<tool_call>{"name": ..., "arguments":
...}</tool_call>`: questo wrapper applica il template passando `tools=` e
riconverte quei blocchi nel formato `AIMessage.tool_calls` atteso da
LangGraph, cosi' il ciclo ReAct in agent/graph.py funziona anche in locale.
"""
from __future__ import annotations

import json
import uuid
from functools import lru_cache
from typing import Any, Sequence

import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _extract_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Estrae ogni blocco <tool_call>{...}</tool_call> emesso dal modello e
    ritorna (tool_calls, testo_restante_senza_quei_blocchi)."""
    tool_calls: list[dict[str, Any]] = []
    remaining = text
    cursor = 0
    while True:
        start = remaining.find(_TOOL_CALL_OPEN, cursor)
        if start == -1:
            break
        end = remaining.find(_TOOL_CALL_CLOSE, start)
        if end == -1:
            break
        payload_text = remaining[start + len(_TOOL_CALL_OPEN) : end].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            cursor = end + len(_TOOL_CALL_CLOSE)
            continue
        tool_calls.append(
            {
                "name": payload.get("name"),
                "args": payload.get("arguments", {}),
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "tool_call",
            }
        )
        remaining = remaining[:start] + remaining[end + len(_TOOL_CALL_CLOSE) :]
    return tool_calls, remaining.strip()


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": str(message.content)}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        entry: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            entry["tool_calls"] = [{"name": tc["name"], "arguments": tc["args"]} for tc in message.tool_calls]
        return entry
    msg = f"Unsupported message type: {type(message)}"
    raise TypeError(msg)


class LocalQwenToolCallingChat(BaseChatModel):
    """Chat model locale (transformers + GPU) con parsing manuale del
    tool-calling secondo il chat template nativo di Qwen2.5."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hf_model: Any
    hf_tokenizer: Any
    max_new_tokens: int = 512
    temperature: float = 0.4

    @property
    def _llm_type(self) -> str:
        return "local-qwen-tool-calling"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        message_dicts = [_message_to_dict(m) for m in messages]
        prompt = self.hf_tokenizer.apply_chat_template(
            message_dicts, tools=tools, tokenize=False, add_generation_prompt=True
        )
        inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)

        with torch.no_grad():
            output_ids = self.hf_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.hf_tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True)

        tool_calls, content = _extract_tool_calls(text)

        message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


@lru_cache(maxsize=1)
def get_chat_model() -> LocalQwenToolCallingChat:
    tokenizer = AutoTokenizer.from_pretrained(config.REASONING_LLM_REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.REASONING_LLM_REPO_ID, dtype=torch.float16, device_map="auto"
    )
    return LocalQwenToolCallingChat(hf_model=model, hf_tokenizer=tokenizer)


if __name__ == "__main__":
    # Debug: `python agent/llm.py` prima verifica _extract_tool_calls con
    # dati finti (istantaneo, nessun modello caricato), poi carica il
    # modello vero e verifica che il tool-calling funzioni end-to-end.
    print("--- test _extract_tool_calls (senza modello) ---")
    fake_text = (
        'Thought: devo cercare.\n<tool_call>\n'
        '{"name": "search_web_tool", "arguments": {"query": "test"}}\n</tool_call>'
    )
    calls, content = _extract_tool_calls(fake_text)
    print("tool_calls:", calls)
    print("content:", repr(content))

    print("\n--- test live: modello + bind_tools (carica il modello su GPU) ---")
    from langchain_core.tools import tool as _tool

    @_tool
    def dummy_search(query: str) -> str:
        """Cerca informazioni su un argomento di viaggio."""
        return f"risultati fittizi per {query}"

    llm = get_chat_model().bind_tools([dummy_search])
    resp = llm.invoke([HumanMessage(content="Use dummy_search to look up 'Bangkok street food'.")])
    print("content:", repr(resp.content[:200]))
    print("tool_calls:", resp.tool_calls)
