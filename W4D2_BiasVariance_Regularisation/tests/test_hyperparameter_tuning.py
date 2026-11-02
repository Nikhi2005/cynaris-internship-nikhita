
"""
W4D2: Automated tests for hyperparameter tuning.
"""

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "hyperparameter_tuning.py"
)


@pytest.fixture(scope="module")
def program_output():
    """Execute the practical script once and capture its output."""

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, (
        f"Program execution failed:\n{result.stderr}"
    )

    return result.stdout


def test_program_executes_successfully(program_output):
    """Check that the program completes all experiment sections."""

    assert "EXPERIMENT COMPLETED" in program_output


def test_wine_dataset_is_loaded(program_output):
    """Check that the Wine dataset has the expected dimensions."""

    assert "Dataset shape: (178, 13)" in program_output


def test_train_test_split_is_correct(program_output):
    """Check the expected training and testing sample counts."""

    assert "Training samples: 142" in program_output
    assert "Testing samples: 36" in program_output


def test_all_search_methods_are_executed(program_output):
    """Check that baseline, Grid Search and Randomized Search run."""

    assert "BASELINE MODEL" in program_output
    assert "GRID SEARCH" in program_output
    assert "RANDOMIZED SEARCH" in program_output


def test_performance_comparison_is_generated(program_output):
    """Check that the final comparison is displayed."""

    assert "PERFORMANCE COMPARISON" in program_output
    assert "Baseline Random Forest" in program_output
    assert "Grid Search" in program_output
    assert "Randomized Search" in program_output
