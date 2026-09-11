from langchain_openai import ChatOpenAI
import config
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START
from typing import Any, Dict, List, Optional, Annotated, TypedDict
from langgraph.graph.message import  BaseMessage
from operator import add
from langchain.messages import SystemMessage, HumanMessage

from RAG.tools import rag_tools


def main():
    llm = ChatOpenAI(
        base_url=config.REMOTE_LLM_URL,
        api_key="ollama",
        model=config.LLM_MODEL_NAME,
        temperature=0,
        default_headers={"ngrok-skip-browser-warning": "true"},
    )

    AGENT_SYSTEM_PROMPT = (
        "You are a knowledgeable AI engineering assistant. When asked about specific "
        "projects, systems, or technical details, you must first call the "
        "`search_knowledge_base` tool to retrieve factual records. Ground your answers "
        "strictly in the returned documents."
    )

    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], add]


    llm_with_tools = llm.bind_tools(rag_tools)
    tools = ToolNode(rag_tools)

    def assistant(state: AgentState): 
        """The assistant function that processes the agent's state and generates a response."""
        messages = state["messages"] + [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}
    
    graph = StateGraph(AgentState)

    graph.add_node('assistant', assistant)
    graph.add_node('tools', tools)

    graph.add_edge(START, 'assistant')
    graph.add_conditional_edges('assistant', tools_condition)
    graph.add_edge('tools', 'assistant')
    workflow = graph.compile()

    ai_message = workflow.invoke({'messages': [HumanMessage(content="Are you able to recreate a similar document, if you're able, try to do so")]})
    print(f"AI Response: {ai_message['messages'][-1].content}")

if __name__ == "__main__":
    main()
