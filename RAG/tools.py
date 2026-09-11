from langchain_core.tools import tool
from RAG.store import get_vector_store
import config

# Initialize retriever once for tool usage
_retriever = get_vector_store().as_retriever(
    search_kwargs={"k": config.TOP_K_RESULTS}
)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal documentation and knowledge base for domain context."""
    results = _retriever.invoke(query)
    if not results:
        return "No relevant internal records found for this query."
    
    return "\n\n---\n\n".join(
        [f"[Context Chunk {i+1}]:\n{doc.page_content}" for i, doc in enumerate(results)]
    )

# List of tools to export to the agent module
rag_tools = [search_knowledge_base]