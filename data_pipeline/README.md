# LangGraph Data Loading, Cleaning & Inspection Pipeline

## Project Overview

This project uses LangGraph and Pandas to load, inspect, clean, and save a rainfall dataset.

## Dataset

- Dataset: Statewise daily rainfall data
- Source file: `rainfall_statewise_daily_imd.csv`
- Rows: 2,201
- Columns: 21

## Workflow

The pipeline follows four sequential steps:

1. **Load:** Reads the rainfall CSV dataset.
2. **Inspect:** Displays the dataset shape, column data types, and missing values.
3. **Clean:** Removes rows with missing dates and converts the `Date` column to datetime format.
4. **Log:** Saves the original and cleaned datasets as CSV artifacts.

## Technologies Used

- Python
- Pandas
- LangGraph

## Project Files

- `main.py` — Implements the LangGraph workflow.
- `requirements.txt` — Lists Python dependencies.
- `raw_rainfall.csv` — Original dataset artifact.
- `cleaned_rainfall.csv` — Cleaned dataset artifact.

## How to Run

1. Install the dependencies:

   `pip install -r requirements.txt`

2. Run the pipeline:

   `python main.py`

## Output

The pipeline displays dataset information and missing-value counts, then generates `raw_rainfall.csv` and `cleaned_rainfall.csv`.

## Current Cleaning Logic

The pipeline removes rows with missing dates and converts valid dates to datetime format. Other missing values are currently reported during inspection but are not removed or imputed.

## Validation

The pipeline was tested successfully with the supplied rainfall dataset.
It loaded 2,201 rows and 21 columns and generated both raw and cleaned CSV artifacts.
