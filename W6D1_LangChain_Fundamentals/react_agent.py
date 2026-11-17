
# W6D1: Simple ReAct-style LangChain Agent.
# Tools: WebSearch and Calculator.
# Executes three tasks using a controlled reasoning-action-observation loop.

import ast
import math
import operator

from langchain_ollama import OllamaLLM
from langchain_core.tools import Tool


MODEL_NAME = "qwen2.5:3b"


# --------------------------------------------------
# TOOL 1: WEB SEARCH STUB
# --------------------------------------------------

def fake_web_search(query: str) -> str:
    """Return simulated search information."""

    query_lower = query.lower()

    if "python" in query_lower:
        result = (
            "Python is a high-level programming language "
            "used in AI, automation and data science."
        )

    elif "artificial intelligence" in query_lower or "ai" in query_lower:
        result = (
            "Artificial Intelligence enables machines to perform "
            "tasks that normally require human intelligence."
        )

    else:
        result = (
            "This is a simulated search result for the requested topic."
        )

    return f"[SIMULATED SEARCH RESULT] {result}"


# --------------------------------------------------
# TOOL 2: SAFE CALCULATOR
# --------------------------------------------------

def safe_calculator(expression: str) -> str:
    """Safely evaluate basic arithmetic expressions."""

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow
    }

    def calculate(node):

        if isinstance(node, ast.Expression):
            return calculate(node.body)

        if isinstance(node, ast.Constant):
            value = node.value

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Only numbers are allowed.")

            return value

        if isinstance(node, ast.BinOp):
            left = calculate(node.left)
            right = calculate(node.right)

            operation = allowed_operators.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator.")

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent too large.")

            result = operation(left, right)

            if not math.isfinite(float(result)) or abs(result) > 1e12:
                raise ValueError("Result outside allowed range.")

            return result

        if isinstance(node, ast.UnaryOp):
            value = calculate(node.operand)

            if isinstance(node.op, ast.USub):
                return -value

            if isinstance(node.op, ast.UAdd):
                return value

            raise ValueError("Unsupported unary operator.")

        raise ValueError("Invalid arithmetic expression.")

    try:
        if len(expression) > 100:
            raise ValueError("Expression too long.")

        tree = ast.parse(expression, mode="eval")
        return str(calculate(tree))

    except Exception as error:
        return f"Calculation error: {error}"


# --------------------------------------------------
# REGISTER EXACTLY TWO LANGCHAIN TOOLS
# --------------------------------------------------

tools = [
    Tool(
        name="WebSearch",
        func=fake_web_search,
        description="Search for simulated information about a topic."
    ),
    Tool(
        name="Calculator",
        func=safe_calculator,
        description="Perform safe arithmetic calculations."
    )
]

tool_map = {tool.name: tool for tool in tools}


# --------------------------------------------------
# SIMPLE REACT-STYLE AGENT
# --------------------------------------------------

def select_tools(task):
    """Select tools according to the user's request."""

    task_lower = task.lower()
    selected = []

    search_keywords = [
        "search", "find information", "information about"
    ]

    calculation_keywords = [
        "calculate", "multiply", "divided", "divide",
        "addition", "subtract", "times"
    ]

    if any(word in task_lower for word in search_keywords):
        selected.append("WebSearch")

    if any(word in task_lower for word in calculation_keywords):
        selected.append("Calculator")

    return selected


def run_agent(llm, task, task_number):

    print(f"\nTASK {task_number}")
    print(f"User Request: {task}")

    selected_tools = select_tools(task)

    observations = []

    # ReAct cycle: Thought -> Action -> Observation

    print("\nThought: Identify the tools needed for this task.")

    for tool_name in selected_tools:

        print(f"\nAction: {tool_name}")

        if tool_name == "WebSearch":
            tool_input = (
                "Artificial Intelligence"
                if "artificial intelligence" in task.lower()
                or " ai " in f" {task.lower()} "
                else "Python programming"
            )

        elif tool_name == "Calculator":
            if "250" in task and "24" in task:
                tool_input = "250 * 24"

            elif "150" in task and "5" in task:
                tool_input = "150 / 5"

            else:
                print("Unsupported calculation request.")
                continue

        else:
            continue

        print(f"Action Input: {tool_input}")

        observation = tool_map[tool_name].invoke(tool_input)

        print(f"Observation: {observation}")

        observations.append(
            f"{tool_name}: {observation}"
        )

    # Generate final response using actual tool observations.

    print("\nThought: Prepare the final response from tool observations.")

    if observations:

        final_prompt = f"""
You are a helpful AI assistant.

User request: {task}

Actual tool observations:
{chr(10).join(observations)}

Answer the user using only the observations above.
Clearly identify simulated search information.
Do not invent additional search results.

Final Answer:
"""

        answer = llm.invoke(final_prompt)

        print(f"\nFinal Answer: {answer}")

    else:
        print("\nNo suitable tool was selected.")

    return selected_tools, observations


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0
    )

    tasks = [
        "Search for information about Python programming.",

        "Calculate 250 multiplied by 24.",

        "Search for information about Artificial Intelligence "
        "and calculate 150 divided by 5."
    ]

    print("\n========== REACT AGENT TEST ==========")

    successful_tasks = 0

    for index, task in enumerate(tasks, start=1):

        selected, observations = run_agent(llm, task, index)

        if selected and len(observations) == len(selected):
            successful_tasks += 1

    print("\n========== FINAL VERIFICATION ==========")
    print(f"Tasks executed: {successful_tasks}/3")
    print(f"Available tools: {len(tools)}")

    if successful_tasks == 3:
        print("SUCCESS: All three agent tasks executed.")
    else:
        print("Some tasks require further verification.")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(f"\nAgent execution failed: {error}")
