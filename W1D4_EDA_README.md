## W1D4 — Exploratory Data Analysis (EDA)

### Objective

Explore the state-wise rainfall dataset using Pandas and visualize numerical distributions, correlations, and state record counts.

### Dataset

- Rows: 2,201
- Columns: 21
- Numerical columns: 8
- String columns: 13
- Exact duplicate rows: 0

### Analysis performed

- Inspected the dataset using `head()`, `describe()`, `info()`, and `isnull().sum()`.
- Identified missing values and potential data-quality issues.
- Generated numerical distribution plots.
- Created a correlation heatmap.
- Plotted the top 10 states by number of records.
- Documented five observations and an EDA narrative.

### Key findings

- `Daily Actual` has a mean of approximately 4.81 and a maximum of 99.2.
- Strong positive correlations include 0.89 between cumulative actual and normal rainfall.
- Category and departure-percentage columns contain missing values.
- Some column names and category values need cleaning.

### Evidence

See `W1D4_EDA_Report.docx` for the detailed observations, EDA narrative, graph screenshots, terminal output evidence, and self-review checklist.

### Files

- `eda_analysis.py` — EDA script
- `numeric_distributions.png` — Numerical distributions
- `correlation_heatmap.png` — Correlation heatmap
- `top10_state_counts.png` — Top 10 state record counts
- `W1D4_EDA_Report.docx` — Complete report
