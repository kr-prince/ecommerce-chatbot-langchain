"""This module contains all the functions which the chat agent uses for running"""

import os
from textwrap import dedent
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from utils.configs import (
    CHATBOT_MODEL_NAME,
    CHATBOT_TEMPERATURE,
    CHATBOT_MAX_TOKENS,
)
from utils.agent_tools import (
    get_intents_from_query,
    get_similar_products_for_order,
    generate_return_authorization,
    get_order_details,
    retrieve_relevant_policies_by_query,
    days_since_date
)


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


class State(TypedDict):
    """State of the chat agent"""
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    """Chat agent assistant"""
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


def load_primary_assistant_prompt():
    """Load the primary assistant prompt"""
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent("""\
                You are a specialised customer support assistant chatbot for a footwear business. You MUST only answer questions related to business.

                ## DECISION MAKING RULES

                Follow the below rules very strictly for doing your job:
                    1. You MUST ALWAYS answer query using only the information provided by your tools.
                    2. If the user is asking for some process related information, use the tool and user query to fetch the policy information, and return the result. Ex - what is the return policy?
                    3. If the user is asking to perform any action given any specific order Id, use the tools to fetch both order details and policy details. Ex - i want to return my order 45673
                    4. In case of any exchange, return or refund be very careful about policy rules and make sure they are not violated. Ex - sale items, days passed since order, etc.
                    5. While checking eligibility for return/exchange etc. be careful about all the policy rules applicable. Always use tool to calculate days passed since order date.
                    6. In case user is asking for product recommendation just use the order Id and the required tool.
                    7. If the user is sure to return the product, first check eligibility, then call the tool to generate request authorization number and send it back to the user.
                    8. Do not create or assume any information. Use the information provided by the tools ONLY. If you cannot answer say you cannot.
                    9. The final answer should be very crisp and to the point in max 2-3 sentences. It also should be conversational and human-like.
                    10. Don't refer the user to chatbot. You are the chatbot and should do the job.
                
                ## END OF RULES
                """)
            ),
            ("placeholder", "{messages}"),
        ]
    )
    return primary_assistant_prompt

def create_and_return_agent_toolbox():
    """Create and agent toolbox"""
    product_recommendor_tool = StructuredTool.from_function(
        func=get_similar_products_for_order,
        name="Product-Recommendor-By-OrderID",
        return_direct=True,
        parse_docstring=True,
        handle_tool_error=True
    )

    return_auth_generator_tool = StructuredTool.from_function(
        func=generate_return_authorization,
        name="Generate-Return-Authorization",
        return_direct=True,
        parse_docstring=True,
        handle_tool_error=True
    )

    get_order_details_tool = StructuredTool.from_function(
        func=get_order_details,
        name="Get-Order-Details",
        return_direct=True,
        parse_docstring=True,
        handle_tool_error=True
    )

    get_relevant_policies_by_query_tool = StructuredTool.from_function(
        func=retrieve_relevant_policies_by_query,
        name="Get-Relevant-Policies-By-Query",
        return_direct=True,
        parse_docstring=True,
        handle_tool_error=True
    )

    days_since_date_tool = StructuredTool.from_function(
        func=days_since_date,
        name="Days-Since-Date",
        return_direct=True,
        parse_docstring=True,
        handle_tool_error=True
    )

    primary_assistant_safe_tools = [
        get_order_details_tool,
        get_relevant_policies_by_query_tool,
        days_since_date_tool,
        product_recommendor_tool,
    ]

    primary_assistant_sensitive_tools = [
        return_auth_generator_tool,
    ]

    primary_assistant_tools = primary_assistant_safe_tools + primary_assistant_sensitive_tools
    return (
        primary_assistant_tools,
        primary_assistant_safe_tools,
        primary_assistant_sensitive_tools,
    )


def create_primary_assistant_runnable_and_build_graph():
    """This function creates the primary assistant runnable and builds the graph"""
    llm = ChatGroq(
        model=CHATBOT_MODEL_NAME,
        temperature=CHATBOT_TEMPERATURE,
        max_tokens=CHATBOT_MAX_TOKENS,
        api_key=os.getenv('GROQ_API_KEY')
    )

    def route_tools(state: State):
        next_node = tools_condition(state)
        # If no tools are invoked, return to the user
        if next_node == END:
            return END
        ai_message = state["messages"][-1]
        # This assumes single tool calls. To handle parallel tool calling, you'd want to
        # use an ANY condition
        first_tool_call = ai_message.tool_calls[0]
        sensitive_tool_names = {t.name for t in primary_assistant_sensitive_tools}
        if first_tool_call["name"] in sensitive_tool_names:
            return "sensitive_tools"
        return "safe_tools"
    
    primary_assistant_prompt = load_primary_assistant_prompt()
    primary_assistant_tools, primary_assistant_safe_tools, \
        primary_assistant_sensitive_tools = create_and_return_agent_toolbox()

    primary_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
        primary_assistant_tools
    )

    builder = StateGraph(State)
    # Define nodes and edges: these do the work
    builder.add_node("assistant", Assistant(primary_assistant_runnable))
    builder.add_node(
        "safe_tools", create_tool_node_with_fallback(primary_assistant_safe_tools)
    )
    builder.add_node(
        "sensitive_tools", create_tool_node_with_fallback(primary_assistant_sensitive_tools)
    )
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant", route_tools, ["safe_tools", "sensitive_tools", END]
    )
    builder.add_edge("safe_tools", "assistant")
    builder.add_edge("sensitive_tools", "assistant")
    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["sensitive_tools"]
    )

    return graph
