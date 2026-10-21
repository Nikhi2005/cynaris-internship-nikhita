import pandas as pd

from W2D3_SMOTE.src.feature_engineering import generate_dataset, create_features


def test_dataset_generation():
    df = generate_dataset()

    assert len(df) == 200
    assert "returned" in df.columns
    assert set(df["returned"].unique()).issubset({0, 1})


def test_feature_creation():
    df = pd.DataFrame({
        "customer_age": [22, 40],
        "order_value": [100.0, 300.0],
        "items_ordered": [2, 3],
        "city_tier": ["Metro", "Tier1"],
        "order_time": pd.to_datetime([
            "2024-01-06 23:00",
            "2024-01-08 10:00",
        ]),
        "returned": [0, 1],
    })

    result = create_features(df)

    assert "city_code" in result.columns
    assert "hour" in result.columns
    assert "is_weekend" in result.columns
    assert "value_per_item" in result.columns
    assert "age_group" in result.columns

    assert result.loc[0, "is_weekend"] == 1
    assert result.loc[0, "is_night"] == 1