"""Plotting and publication-quality visualization module for VoxShield."""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import FIGURES_DIR

logger = get_logger("voxshield.reporting.plots")

# Set standard publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})


def plot_class_counts(df: pd.DataFrame, output_path: Path = FIGURES_DIR / "class_distribution.png") -> Path:
    """Plots the overall class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    palette = {"scam": "#d9534f", "legitimate": "#5cb85c"}
    counts = df["label"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=palette, hue=counts.index, legend=False)
    ax.set_title("VISHGUARD Class Distribution")
    ax.set_xlabel("Call Category")
    ax.set_ylabel("Number of Samples")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", weight="bold", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved class distribution plot to {output_path}")
    return output_path


def plot_language_class_distribution(df: pd.DataFrame, output_path: Path = FIGURES_DIR / "language_class_distribution.png") -> Path:
    """Plots class balance across languages."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    palette = {"scam": "#d9534f", "legitimate": "#5cb85c"}
    order = sorted(df["language"].unique())
    sns.countplot(data=df, x="language", hue="label", palette=palette, order=order, ax=ax)
    ax.set_title("Class Distribution Across Languages (English, French, Arabic)")
    ax.set_xlabel("Language Code")
    ax.set_ylabel("Count")
    ax.legend(title="Class", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved language class distribution plot to {output_path}")
    return output_path


def plot_duration_distribution(df: pd.DataFrame, output_path: Path = FIGURES_DIR / "audio_duration_distribution.png") -> Path:
    """Plots audio duration distribution split by scam vs legitimate."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    palette = {"scam": "#d9534f", "legitimate": "#5cb85c"}
    sns.histplot(data=df, x="duration", hue="label", palette=palette, kde=True, bins=25, ax=ax, alpha=0.6)
    ax.set_title("Audio Duration Distribution by Class")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Call Count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved duration distribution plot to {output_path}")
    return output_path


def plot_transcript_length_distribution(df: pd.DataFrame, output_path: Path = FIGURES_DIR / "transcript_length_distribution.png") -> Path:
    """Plots transcript word count distribution split by scam vs legitimate."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    palette = {"scam": "#d9534f", "legitimate": "#5cb85c"}
    sns.histplot(data=df, x="words_count", hue="label", palette=palette, kde=True, bins=25, ax=ax, alpha=0.6)
    ax.set_title("Transcript Word Count Distribution by Class")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Call Count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved transcript length distribution plot to {output_path}")
    return output_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["legitimate", "scam"],
    output_path: Path = FIGURES_DIR / "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> Path:
    """Plots and saves a normalized confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(cm_norm, annot=annotations, fmt="", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {output_path}")
    return output_path


def plot_model_comparisons(
    metrics_df: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "model_comparison.png",
) -> Path:
    """Plots F1, Precision, Recall, and ROC-AUC comparison across models."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    metrics_to_plot = ["accuracy", "precision_scam", "recall_scam", "f1_scam", "roc_auc"]
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    melted = metrics_df.melt(id_vars=["experiment_id"], value_vars=available_metrics,
                             var_name="Metric", value_name="Score")
    
    sns.barplot(data=melted, x="experiment_id", y="Score", hue="Metric", ax=ax, palette="Set2")
    ax.set_title("Cross-Modality Model Performance Comparison (Held-Out Test Set)")
    ax.set_xlabel("Experiment ID")
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved model comparison plot to {output_path}")
    return output_path
