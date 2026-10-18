from pathlib import Path

# Locate the Day 5 project directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def test_loss_plot_exists():
    """Check that the training dashboard image exists."""
    assert (OUTPUT_DIR / "loss_plot.png").is_file()


def test_cat_plot_exists():
    """Check that the categorical plot image exists."""
    assert (OUTPUT_DIR / "cat_plot.png").is_file()