# W1D5: Data Visualisation — Matplotlib & Seaborn

## Overview

This project demonstrates data visualisation using Matplotlib and Seaborn.
It generates simulated model-training curves and a categorical bar plot.

## Features

- Simulates training and validation loss across 20 epochs.
- Simulates training and validation accuracy across 20 epochs.
- Creates a Matplotlib training dashboard.
- Creates a Seaborn categorical bar plot.
- Saves both plots as PNG images.
- Includes automated tests using pytest.

## Project Structure

- `src/visualisation.py` — data generation and plotting code.
- `outputs/loss_plot.png` — training and validation dashboard.
- `outputs/cat_plot.png` — categorical bar plot.
- `tests/test_visualisation.py` — automated output-file tests.

## Requirements

- Python
- NumPy
- Matplotlib
- Seaborn
- pytest

## Run the Program

```powershell
python src/visualisation.py
```
