import os
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_dartmouth.llms import ChatDartmouthCloud
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Initialize language models / agents
researcher_llm = ChatDartmouthCloud(
    model_name="openai.gpt-4.1-mini-2025-04-14",
    inference_server_url="https://chat-dev.dartmouth.edu/api/",
    dartmouth_chat_api_key=os.environ["DARTMOUTH_CHAT_DEV_API_KEY"],
)
researcher_llm = create_react_agent(model=researcher_llm, tools=[wikipedia])


writer_llm = ChatDartmouthCloud(
    model_name="openai.gpt-4.1-mini-2025-04-14",
    inference_server_url="https://chat-dev.dartmouth.edu/api/",
    dartmouth_chat_api_key=os.environ["DARTMOUTH_CHAT_DEV_API_KEY"],
)


# Define the state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The conversation history"]


# Define agent nodes
def researcher_agent(state: AgentState) -> AgentState:
    """Researcher agent finds information on a topic"""
    messages = state["messages"]

    # Extract the query from the last human message
    human_message = messages[0]

    # Craft a researcher prompt
    researcher_prompt = f"""You are a research assistant. Find key information about the topic: {human_message.content}
    Be thorough but concise. Focus on facts and important details that would be needed for a summary.
    """

    # Generate researcher response
    response = researcher_llm.invoke(
        {"messages": [HumanMessage(content=researcher_prompt)]}
    )

    return {"messages": response["messages"]}


def writer_agent(state: AgentState) -> AgentState:
    """Writer agent summarizes the information"""
    messages = state["messages"]

    research_message = messages[-1]

    # Craft a writer prompt
    writer_prompt = f"""You are a skilled writer of educational children's science materials. Based on the following research, create a well-structured summary for the target audience of 6 to 8 year old kids.

    {research_message.content}

    Make it engaging, clear, and concise.
    """

    # Generate writer response
    response = writer_llm.invoke([HumanMessage(content=writer_prompt)])

    # Add response to messages
    new_messages = messages + [AIMessage(content=response.content, name="writer")]

    return {"messages": new_messages}


# Create the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)

# Add the edges
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# Set the entry point
workflow.set_entry_point("researcher")

# Compile the graph
graph = workflow.compile()

print(graph.get_graph().draw_ascii())

initial_state = {
    "messages": [HumanMessage(content="What is Dartmouth College?")],
    "next": "researcher",
}

# Run the graph
result = graph.invoke(initial_state)

# Print the final result
for message in result["messages"]:
    message.pretty_print()
