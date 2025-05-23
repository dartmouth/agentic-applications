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


tools = [multiply]

# Create the agent
memory = MemorySaver()
model = ChatDartmouthCloud(
    model_name="openai.gpt-4.1-mini-2025-04-14",
    inference_server_url="https://chat-dev.dartmouth.edu/api/",
    dartmouth_chat_api_key=os.environ["DARTMOUTH_CHAT_DEV_API_KEY"],
)

# model = model.bind_tools(tools)

messages = [HumanMessage("What is 3 times 12?")]
response = model.invoke(messages)
response.pretty_print()
# messages.append(response)

# if tool_calls := response.tool_calls:
#     for tool_call in tool_calls:
#         fn, *_ = [tool for tool in tools if tool.name == tool_call["name"]]
#         tool_msg = fn.invoke(tool_call)
#         messages.append(tool_msg)

# print(model.invoke(messages))

# agent_executor = create_react_agent(model, tools, checkpointer=memory)

# config = {"configurable": {"thread_id": "abc123"}}
# response = agent_executor.invoke({"messages": messages}, config)

# for message in response["messages"]:
#     message.pretty_print()
