from tools.finetuned_draft_tool import draft_with_finetuned_model
from tools.kg_tools import kg_query_tool, update_knowledge_graph
from tools.quality_tool import resource_quality_tool
from tools.rag_tool import rag_retrieval_tool
from tools.search_tool import search_web_tool

# Tool esposti al ciclo ReAct dell'agente (bind_tools).
REACT_TOOLS = [
    search_web_tool,
    rag_retrieval_tool,
    kg_query_tool,
    resource_quality_tool,
    draft_with_finetuned_model,
]

__all__ = [
    "search_web_tool",
    "rag_retrieval_tool",
    "kg_query_tool",
    "resource_quality_tool",
    "draft_with_finetuned_model",
    "update_knowledge_graph",
    "REACT_TOOLS",
]
