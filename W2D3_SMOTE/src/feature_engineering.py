import pandas as pd
import numpy as np
from pathlib import Path


# Project paths
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "synthetic_ecom.csv"


def generate_dataset():
    """Generate and save a reproducible e-commerce dataset."""
    np.random.seed(42)

    n = 200

    df = pd.DataFrame({
        "customer_age": np.random.randint(18, 70, n),
        "order_value": np.random.exponential(scale=150, size=n).round(2),
        "items_ordered": np.random.randint(1, 15, n),
        "city_tier": np.random.choice(
            ["Metro", "Tier1", "Tier2", "Tier3"], n
        ),
        "order_time": pd.date_range(
            "2024-01-01", periods=n, freq="3h"
        ),
        "returned": np.random.choice(
            [0, 1], n, p=[0.85, 0.15]
        ),
    })

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)

    print(f"Dataset saved to: {DATA_FILE}")
    print(f"Dataset shape: {df.shape}")
    print("Return counts:")
    print(df["returned"].value_counts())

    return df


def create_features(df):
    """Create useful features from the e-commerce dataset."""
    df = df.copy()

    # Encode city categories
    tier_map = {
        "Metro": 0,
        "Tier1": 1,
        "Tier2": 2,
        "Tier3": 3,
    }
    df["city_code"] = df["city_tier"].map(tier_map)

    # Extract date and time features
    df["hour"] = df["order_time"].dt.hour
    df["day_of_week"] = df["order_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = (
        (df["hour"] < 6) | (df["hour"] >= 22)
    ).astype(int)

    # Create additional order features
    df["value_per_item"] = (
        df["order_value"] / df["items_ordered"]
    ).round(2)

    df["high_value_order"] = (
        df["order_value"] > df["order_value"].quantile(0.75)
    ).astype(int)

    # Group customers by age
    df["age_group"] = pd.cut(
        df["customer_age"],
        bins=[0, 25, 35, 50, 100],
        labels=["Young", "Prime", "Senior", "Elder"],
    )

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    engineered_df = create_features(dataset)

    print("\n=== Engineered Features ===")
    print(
        engineered_df[
            [
                "customer_age",
                "age_group",
                "order_value",
                "high_value_order",
                "is_weekend",
                "value_per_item",
                "returned",
            ]
        ].head(10)
    )

    print("\n=== Return Rate by Weekend ===")
    print(
        engineered_df.groupby("is_weekend")["returned"]
        .mean()
        .round(3)
    )

    print("\n=== Return Rate by Age Group ===")
    print(
        engineered_df.groupby("age_group", observed=False)["returned"]
        .mean()
        .round(3)
    )