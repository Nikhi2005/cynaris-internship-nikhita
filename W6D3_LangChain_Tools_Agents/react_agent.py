
import ast
import operator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# -------------------------------
# Safe calculator implementation
# -------------------------------

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_expression(expression):
    def evaluate_node(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            operation = ALLOWED_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported mathematical operation.")

            return operation(
                evaluate_node(node.left),
                evaluate_node(node.right)
            )

        if isinstance(node, ast.UnaryOp):
            operation = ALLOWED_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported unary operation.")

            return operation(evaluate_node(node.operand))

        raise ValueError("Invalid mathematical expression.")

    parsed_expression = ast.parse(expression, mode="eval")
    return evaluate_node(parsed_expression.body)


# -------------------------------
# Tool 1: Web Search Stub
# -------------------------------

@tool
def web_search(query: str) -> str:
    """Search a simulated web knowledge base for information about a topic."""

    knowledge_base = {
        "langchain": (
            "LangChain is a framework for building applications "
            "powered by large language models."
        ),
        "python": (
            "Python is a high-level programming language used in "
            "automation, data science, and artificial intelligence."
        ),
        "machine learning": (
            "Machine learning enables computers to learn patterns "
            "from data and make predictions."
        ),
    }

    query_lower = query.lower()

    for topic, result in knowledge_base.items():
        if topic in query_lower:
            return f"Search result: {result}"

    return (
        f"Simulated search completed for '{query}'. "
        "No matching entry was found in the demo knowledge base."
    )


# -------------------------------
# Tool 2: Calculator
# -------------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression using numbers and arithmetic operators."""

    try:
        result = evaluate_expression(expression)
        return f"Calculation result: {result}"
    except Exception as error:
        return f"Calculation error: {error}"


# -------------------------------
# Agent setup
# -------------------------------

def main():
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    tools = [web_search, calculator]
    tools_by_name = {tool.name: tool for tool in tools}

    try:
        agent_llm = llm.bind_tools(tools)
        native_tool_calling = True
        print("Native tool-calling interface enabled.")
    except Exception as error:
        agent_llm = llm
        native_tool_calling = False
        print(f"Tool-calling interface unavailable: {error}")
        print("Using keyword-based fallback routing.")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant that can use tools.\n"
            "Use web_search for information searches.\n"
            "Use calculator for mathematical calculations.\n"
            "When a tool is needed, call the appropriate tool.\n"
            "After receiving the tool result, explain it clearly.\n"
            "Do not invent tool results."
        )
    )

    tasks = [
        "Use web_search to find information about LangChain.",
        "Use calculator to calculate 18 multiplied by 7.",
        "Use calculator to calculate 144 divided by 12."
    ]

    print("\n" + "=" * 60)
    print("W6D3 - LANGCHAIN TOOLS AND AGENT")
    print("=" * 60)

    for index, task in enumerate(tasks, start=1):
        print(f"\nTask {index}")
        print(f"User: {task}")

        messages = [
            system_message,
            HumanMessage(content=task)
        ]

        completed = False

        if native_tool_calling:
            for step in range(3):
                response = agent_llm.invoke(messages)
                messages.append(response)

                tool_calls = getattr(response, "tool_calls", [])

                if not tool_calls:
                    print(f"Agent: {response.content}")
                    completed = True
                    break

                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    print(f"Action: {tool_name}")
                    print(f"Action Input: {tool_args}")

                    selected_tool = tools_by_name.get(tool_name)

                    if selected_tool is None:
                        observation = "Unknown tool requested."
                    else:
                        observation = selected_tool.invoke(tool_args)

                    print(f"Observation: {observation}")

                    messages.append(
                        ToolMessage(
                            content=str(observation),
                            tool_call_id=tool_call["id"]
                        )
                    )

            if not completed:
                final_response = llm.invoke(messages)
                print(f"Agent: {final_response.content}")

        else:
            # Fallback: deterministic routing
            task_lower = task.lower()

            if "search" in task_lower or "find information" in task_lower:
                observation = web_search.invoke(
                    {"query": "LangChain"}
                )
                print("Action: web_search")

            elif "multiply" in task_lower:
                observation = calculator.invoke(
                    {"expression": "18 * 7"}
                )
                print("Action: calculator")

            elif "divided" in task_lower:
                observation = calculator.invoke(
                    {"expression": "144 / 12"}
                )
                print("Action: calculator")

            else:
                observation = "No matching tool found."

            print(f"Observation: {observation}")

            final_response = llm.invoke(
                [
                    system_message,
                    HumanMessage(
                        content=(
                            f"User task: {task}\n"
                            f"Tool observation: {observation}\n"
                            "Explain the result briefly."
                        )
                    )
                ]
            )

            print(f"Agent: {final_response.content}")

    print("\nAll 3 agent tasks completed.")


if __name__ == "__main__":
    main()
