
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

np.random.seed(42)
epochs = list(range(1, 21))

train_loss = [
    2.5 * np.exp(-0.15 * e) + 0.05 * np.random.randn()
    for e in epochs
]

val_loss = [
    2.8 * np.exp(-0.12 * e) + 0.1 * np.random.randn()
    for e in epochs
]

train_acc = [
    1 - 0.9 * np.exp(-0.2 * e) + 0.01 * np.random.randn()
    for e in epochs
]

val_acc = [
    1 - 0.95 * np.exp(-0.18 * e) + 0.015 * np.random.randn()
    for e in epochs
]

print("Number of epochs:", len(epochs))
print("Training loss values:", len(train_loss))
print("Validation loss values:", len(val_loss))
print("Training accuracy values:", len(train_acc))
print("Validation accuracy values:", len(val_acc))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

fig.suptitle(
    "Model Training Dashboard",
    fontsize=13,
    fontweight="bold"
)


ax1.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)
ax1.plot(epochs, val_loss, "r-s", label="Val Loss", markersize=4)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")

ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs, train_acc, "b-o", label="Train Acc", markersize=4)
ax2.plot(epochs, val_acc, "r-s", label="Val Acc", markersize=4)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training & Validation Accuracy")

ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
output_dir = Path(__file__).resolve().parent.parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "loss_plot.png", dpi=80)

print("Training dashboard saved to outputs/loss_plot.png")
print(f"Final Train Acc: {train_acc[-1]:.3f}")
print(f"Final Val Acc:   {val_acc[-1]:.3f}")


# Seaborn
subjects = ["Mathematics", "Science", "English"]
average_scores = [82, 76, 88]

plt.figure(figsize=(8, 5))

sns.barplot(x=subjects, y=average_scores)

plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.title("Average Score by Subject")
plt.tight_layout()

output_dir = Path(__file__).resolve().parent.parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "cat_plot.png", dpi=100)
plt.close()

print("Categorical plot saved to outputs/cat_plot.png")