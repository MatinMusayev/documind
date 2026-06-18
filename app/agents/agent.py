import os
from typing import Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from app.core.rag_pipeline import rag_query, retrieve_context


def document_search_tool(query: str) -> str:
    result = rag_query(query)
    return f"Answer: {result['answer']}\nSources: {', '.join(result['sources'])}"


def summarize_tool(query: str) -> str:
    contexts = retrieve_context(query, n_results=8)
    combined = "\n".join(c["text"] for c in contexts)
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following document excerpts concisely."},
            {"role": "user", "content": combined}
        ],
        max_tokens=512
    )
    return response.choices[0].message.content


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    tools = [
        Tool(
            name="DocumentSearch",
            func=document_search_tool,
            description="Search uploaded documents for specific information. Input: a question or topic."
        ),
        Tool(
            name="DocumentSummarizer",
            func=summarize_tool,
            description="Summarize content from uploaded documents on a topic. Input: topic or document name."
        ),
    ]

    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=3))

    prompt = PromptTemplate.from_template(
        """You are DocuMind Agent, an AI research assistant with access to tools.

Available tools: {tools}
Tool names: {tool_names}

Use this format:
Question: the input question
Thought: think about what to do
Action: tool name
Action Input: input to the tool
Observation: result
... (repeat as needed)
Thought: I now know the final answer
Final Answer: your final answer

Question: {input}
{agent_scratchpad}"""
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5, handle_parsing_errors=True)


def run_agent(query: str) -> Dict[str, Any]:
    agent_executor = build_agent()
    try:
        result = agent_executor.invoke({"input": query})
        return {"answer": result.get("output", "No answer"), "mode": "agent"}
    except Exception as e:
        return {"answer": f"Agent error: {str(e)}", "mode": "agent"}
