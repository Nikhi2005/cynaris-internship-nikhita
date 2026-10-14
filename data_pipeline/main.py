import pandas as pd
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class PipelineState(TypedDict):
    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame

def load_data(state: PipelineState):
    data = pd.read_csv(
    "data/rainfall_statewise_daily_imd.csv"
)

    print("Dataset loaded successfully!")
    print("Shape:", data.shape)

    return {"raw_data": data}

def inspect_data(state: PipelineState):
    data = state["raw_data"]

    print("\nDataset Information:")
    print("Shape:", data.shape)
    print("\nData Types:")
    print(data.dtypes)
    print("\nMissing Values:")
    print(data.isnull().sum())

    return {}

def clean_data(state: PipelineState):
    data = state["raw_data"].copy()

    data = data.dropna(subset=["Date"])
    data["Date"] = pd.to_datetime(data["Date"])

    print("\nCleaning completed!")
    print("Cleaned shape:", data.shape)

    return {"cleaned_data": data}


def log_results(state: PipelineState):
    raw_data = state["raw_data"]
    cleaned_data = state["cleaned_data"]

    raw_data.to_csv("raw_rainfall.csv", index=False)
    cleaned_data.to_csv("cleaned_rainfall.csv", index=False)

    print("\nPipeline results saved!")
    print("Raw dataset: raw_rainfall.csv")
    print("Cleaned dataset: cleaned_rainfall.csv")
    print("Original rows:", len(raw_data))
    print("Rows after cleaning:", len(cleaned_data))

    return {}

# Create the workflow
builder = StateGraph(PipelineState)

builder.add_node("load", load_data)
builder.add_node("inspect", inspect_data)
builder.add_node("clean", clean_data)
builder.add_node("log", log_results)

builder.add_edge(START, "load")
builder.add_edge("load", "inspect")
builder.add_edge("inspect", "clean")
builder.add_edge("clean", "log")
builder.add_edge("log", END)

graph = builder.compile()

result = graph.invoke({
    "raw_data": pd.DataFrame(),
    "cleaned_data": pd.DataFrame()
})