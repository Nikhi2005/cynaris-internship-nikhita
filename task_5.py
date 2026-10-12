
import os
import pandas as pd
import numpy as np


def main():
    # ---------- Load CSV dataset ----------
    csv_path = os.path.join("data", "student_scores.csv")
    df = pd.read_csv(csv_path)

    print("Dataset (first 5 rows):\n", df.head())
    print(f"\nDataset shape: {df.shape}")

    # ---------- Handle missing values ----------
    df_clean = df.dropna(subset=["study_hours", "exam_score"])

    # ---------- Convert selected columns to NumPy ----------
    data = df_clean[["study_hours", "exam_score"]].to_numpy()

    # ---------- Calculate mean ----------
    mean_values = np.mean(data, axis=0)
    print("\nMean (study hours, exam score):", mean_values)

    # ---------- Calculate standard deviation ----------
    std_values = np.std(data, axis=0, ddof=1)
    print("Sample standard deviation:", std_values)

    # ---------- Calculate correlation ----------
    correlation = np.corrcoef(data[:, 0], data[:, 1])
    print("Correlation matrix:\n", correlation)


if __name__ == "__main__":
    main()
