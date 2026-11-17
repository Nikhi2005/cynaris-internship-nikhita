
# W6D2: Tool-based agent with simulated search and safe calculator

import ast
import math
import operator

from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM

MODEL_NAME = "qwen2.5:3b"


# Allowed arithmetic operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow
}


def safe_calculator(expression):
    """Evaluate basic arithmetic without unrestricted eval()."""

    if len(expression) > 100:
        return "Error: Expression is too long."

    def calculate(node):

        if isinstance(node, ast.Expression):
            return calculate(node.body)

        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.UnaryOp):
            value = calculate(node.operand)

            if isinstance(node.op, ast.UAdd):
                return value

            if isinstance(node.op, ast.USub):
                return -value

            raise ValueError("Unsupported unary operator.")

        if isinstance(node, ast.BinOp):

            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Unsupported arithmetic operator.")

            left = calculate(node.left)
            right = calculate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            result = ALLOWED_OPERATORS[type(node.op)](left, right)

            if not math.isfinite(result) or abs(result) > 1e12:
                raise ValueError("Result is outside the allowed range.")

            return result

        raise ValueError("Invalid arithmetic expression.")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(calculate(tree))

    except (SyntaxError, ValueError, TypeError,
            ZeroDivisionError, OverflowError) as error:
        return f"Calculation error: {error}"


def fake_web_search(query):
    """Return predefined information, not live internet results."""

    knowledge = {
        "langchain": (
            "LangChain is a framework for developing applications "
            "powered by language models. It provides components "
            "such as prompts, chains, memory and tools."
        ),

        "conversation memory": (
            "Conversation memory stores previous user and assistant "
            "messages so a chatbot can use earlier context."
        )
    }

    query = query.lower()

    for topic, information in knowledge.items():
        if topic in query:
            return "[SIMULATED SEARCH RESULT] " + information

    return "[SIMULATED SEARCH RESULT] No matching information found."


# Create exactly two LangChain tools
tools = [
    Tool(
        name="WebSearch",
        func=fake_web_search,
        description="Returns predefined information about LangChain and conversation memory."
    ),

    Tool(
        name="Calculator",
        func=safe_calculator,
        description="Performs safe arithmetic calculations."
    )
]

tool_map = {
    tool.name: tool for tool in tools
}


def run_agent(task, llm):
    """Execute the required tools and generate a final response."""

    print("\n" + "-" * 55)
    print("Task:", task)

    observations = []

    task_lower = task.lower()

    # Select WebSearch for LangChain-related information
    if "langchain" in task_lower:
        result = tool_map["WebSearch"].invoke("LangChain")
        observations.append(("WebSearch", result))

    # Select WebSearch for conversation memory
    if "conversation memory" in task_lower:
        result = tool_map["WebSearch"].invoke("conversation memory")
        observations.append(("WebSearch", result))

    # Select Calculator for 18 multiplied by 7
    if "18" in task_lower and "7" in task_lower:
        result = tool_map["Calculator"].invoke("18 * 7")
        observations.append(("Calculator", result))

    # Select Calculator for 144 divided by 12
    if "144" in task_lower and "12" in task_lower:
        result = tool_map["Calculator"].invoke("144 / 12")
        observations.append(("Calculator", result))

    if not observations:
        print("No supported tool action identified.")
        return

    print("\nTool Execution:")

    for name, observation in observations:
        print("Action:", name)
        print("Observation:", observation)

    observation_text = "\n".join(
        f"{name}: {result}" for name, result in observations
    )

    final_prompt = f"""
You are an AI assistant.
Answer the task using only the observations below.
Do not invent information.

Task: {task}

Observations:
{observation_text}

Final Answer:
"""

    final_answer = llm.invoke(final_prompt)

    print("\nFinal Answer:", final_answer)


def main():

    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.2
    )

    tasks = [
        "Search for information about LangChain.",
        "Calculate 18 multiplied by 7.",
        "Explain conversation memory and calculate 144 divided by 12."
    ]

    print("=" * 55)
    print("W6D2 - TWO TOOL AGENT")
    print("=" * 55)

    print("Available tools:", len(tools))

    for task in tasks:
        run_agent(task, llm)

    print("\nTasks processed:", len(tasks), "/", len(tasks))
    print("Agent execution completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Agent execution failed:", error)
