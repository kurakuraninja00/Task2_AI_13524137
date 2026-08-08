"""
visualization.py — Visualisasi: tree structure, loss contour, parameter trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List
import os
from . import config


def plot_tree_structure(
    tree_dict: dict,
    feature_names: List[str] = None,
    max_depth_display: int = 4,
    save_path: str = None,
    figsize: tuple = (24, 14),
) -> None:
    """
    Visualisasi struktur tree sebagai gambar menggunakan matplotlib.

    Parameters
    ----------
    tree_dict : dict dari DecisionTreeCART.get_tree_structure()
    feature_names : nama fitur untuk label
    max_depth_display : kedalaman maksimum yang ditampilkan
    save_path : path untuk simpan gambar
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("CART Decision Tree Structure (from scratch)", fontsize=16, fontweight="bold")

    def _draw_node(node_dict, x, y, x_range, depth):
        if depth > max_depth_display:
            return

        # Node box
        if node_dict.get("is_leaf", False):
            color = "#90EE90" if node_dict["prediction"] == 1 else "#FFB6C1"
            label = f"Leaf\nPred: {node_dict['prediction']}\n"
            label += f"Dist: {node_dict['class_dist']}\n"
            label += f"n={node_dict['n_samples']}"
        else:
            color = "#ADD8E6"
            feat = node_dict.get("feature_idx", "?")
            if feature_names and isinstance(feat, int) and feat < len(feature_names):
                feat_name = feature_names[feat]
            else:
                feat_name = f"X[{feat}]"
            label = f"{feat_name}\n≤ {node_dict.get('threshold', '?')}\n"
            label += f"gini={node_dict['gini']}\n"
            label += f"n={node_dict['n_samples']}"

        bbox = dict(
            boxstyle="round,pad=0.3",
            facecolor=color,
            edgecolor="black",
            linewidth=1,
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=7,
                bbox=bbox, transform=ax.transAxes)

        # Draw children
        if not node_dict.get("is_leaf", False) and depth < max_depth_display:
            dx = x_range / 4
            y_step = 0.85 / (max_depth_display + 1)

            # Left child
            if "left" in node_dict:
                lx = x - dx
                ly = y - y_step
                ax.plot([x, lx], [y - 0.02, ly + 0.02], "k-", linewidth=0.8,
                        transform=ax.transAxes)
                ax.text((x + lx) / 2 - 0.01, (y + ly) / 2, "True",
                        fontsize=6, color="green", transform=ax.transAxes)
                _draw_node(node_dict["left"], lx, ly, x_range / 2, depth + 1)

            # Right child
            if "right" in node_dict:
                rx = x + dx
                ry = y - y_step
                ax.plot([x, rx], [y - 0.02, ry + 0.02], "k-", linewidth=0.8,
                        transform=ax.transAxes)
                ax.text((x + rx) / 2 + 0.01, (y + ry) / 2, "False",
                        fontsize=6, color="red", transform=ax.transAxes)
                _draw_node(node_dict["right"], rx, ry, x_range / 2, depth + 1)

    _draw_node(tree_dict, 0.5, 0.95, 1.0, 0)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Tree saved to {save_path}")
    plt.show()


def plot_loss_curves(
    loss_histories: dict,
    title: str = "Loss Convergence",
    save_path: str = None,
) -> None:
    """
    Plot kurva loss vs iterasi untuk beberapa optimizer.

    Parameters
    ----------
    loss_histories : dict {nama_optimizer: list_of_losses}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, losses in loss_histories.items():
        ax.plot(losses, label=name, linewidth=2)
    ax.set_xlabel("Iteration (×10)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_loss_contour_and_trajectory(
    X: np.ndarray,
    y: np.ndarray,
    w_history: List[np.ndarray],
    sample_weights: np.ndarray = None,
    dim1: int = 0,
    dim2: int = 1,
    feature_names: List[str] = None,
    title: str = "Loss Contour & Parameter Trajectory",
    save_path: str = None,
    lambda_reg: float = 0.01,
) -> None:
    """
    Visualisasi kontur loss 2D dan lintasan parameter selama training.

    Proyeksi ke 2 dimensi parameter terpilih.
    """
    if sample_weights is None:
        sample_weights = np.ones(len(y))

    # Extract 2D trajectory
    w_traj = np.array(w_history)
    w1_traj = w_traj[:, dim1]
    w2_traj = w_traj[:, dim2]

    # Create grid around trajectory
    margin = 0.5
    w1_min, w1_max = w1_traj.min() - margin, w1_traj.max() + margin
    w2_min, w2_max = w2_traj.min() - margin, w2_traj.max() + margin

    w1_grid = np.linspace(w1_min, w1_max, 80)
    w2_grid = np.linspace(w2_min, w2_max, 80)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)

    # Final weights sebagai base
    w_final = w_traj[-1].copy()

    # Compute loss pada setiap grid point
    Z = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_test = w_final.copy()
            w_test[dim1] = W1[i, j]
            w_test[dim2] = W2[i, j]
            # BCE loss
            z_val = X @ w_test
            z_val = np.clip(z_val, -500, 500)
            y_hat = 1.0 / (1.0 + np.exp(-z_val))
            y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
            bce = -sample_weights * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            Z[i, j] = bce.mean() + 0.5 * lambda_reg * np.sum(w_test ** 2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(W1, W2, Z, levels=30, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="Loss")

    # Trajectory
    ax.plot(w1_traj, w2_traj, "r.-", linewidth=1.5, markersize=4, alpha=0.7, label="Parameter trajectory")
    ax.plot(w1_traj[0], w2_traj[0], "go", markersize=10, label="Start", zorder=5)
    ax.plot(w1_traj[-1], w2_traj[-1], "r*", markersize=15, label="End", zorder=5)

    dim1_name = feature_names[dim1] if feature_names and dim1 < len(feature_names) else f"w[{dim1}]"
    dim2_name = feature_names[dim2] if feature_names and dim2 < len(feature_names) else f"w[{dim2}]"

    ax.set_xlabel(dim1_name, fontsize=12)
    ax.set_ylabel(dim2_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison_table(
    results: dict,
    title: str = "Model Comparison: From-Scratch vs Sklearn",
    save_path: str = None,
) -> None:
    """
    Plot tabel perbandingan macro F1 from-scratch vs sklearn.

    Parameters
    ----------
    results : dict {model_name: {"scratch": f1, "sklearn": f1}}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    models = list(results.keys())
    cell_text = []
    for m in models:
        scratch = results[m].get("scratch", "-")
        sklearn_val = results[m].get("sklearn", "-")
        if isinstance(scratch, float):
            scratch = f"{scratch:.4f}"
        if isinstance(sklearn_val, float):
            sklearn_val = f"{sklearn_val:.4f}"
        cell_text.append([m, scratch, sklearn_val])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Model", "From-Scratch (Macro F1)", "Sklearn (Macro F1)"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
