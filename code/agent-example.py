from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from langchain_dartmouth.llms import ChatDartmouthCloud
import os

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def manual_agent():
    """Manually going through the tool calling steps"""

    tools = [multiply]
    # Create the agent
    model = ChatDartmouthCloud(
        model_name="openai.gpt-4.1-mini-2025-04-14",
    )

    model = model.bind_tools(tools)

    messages = [HumanMessage("What is 3 times 12?")]
    response = model.invoke(messages)
    messages.append(response)

    if tool_calls := response.tool_calls:
        for tool_call in tool_calls:
            fn, *_ = [tool for tool in tools if tool.name == tool_call["name"]]
            tool_msg = fn.invoke(tool_call)
            messages.append(tool_msg)
    response = model.invoke(messages)
    messages.append(response)

    for message in messages:
        message.pretty_print()


def automated_agent():
    """Automating the function/tool calling step"""

    tools = [multiply]

    # Create the agent
    model = ChatDartmouthCloud(
        model_name="openai.gpt-4.1-mini-2025-04-14",
    )

    model = model.bind_tools(tools)

    # Create the executor
    agent_executor = create_react_agent(model, tools)

    # Send a message
    messages = [HumanMessage("What is 3 times 12?")]
    config = {"configurable": {"thread_id": "abc123"}}
    response = agent_executor.invoke({"messages": messages}, config)

    for message in response["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    manual_agent()
    automated_agent()
