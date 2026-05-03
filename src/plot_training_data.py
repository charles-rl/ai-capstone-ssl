import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use("science")
import os
import re
import numpy as np


def load_and_clean_csv(filepath):
    """Read CSV, drop WandB exported min/max/_step columns, return df and x-column name."""
    df = pd.read_csv(filepath)

    x_col = None
    for candidate in ("Step", "epoch"):
        if candidate in df.columns:
            x_col = candidate
            break
    if x_col is None:
        for c in df.columns:
            if c.lower() in ("step", "epoch"):
                x_col = c
                break

    drop_cols =[c for c in df.columns if ("__MIN" in c) or ("__MAX" in c) or ("_step" in c)]
    if x_col in drop_cols:
        drop_cols.remove(x_col)

    df_clean = df.drop(columns=drop_cols, errors="ignore")

    if x_col is None:
        for c in df_clean.columns:
            if np.issubdtype(df_clean[c].dtype, np.number):
                x_col = c
                break
    return df_clean, x_col


def _clean_temp_label(col):
    col_l = col.lower()
    if "0.1" in col_l:
        return r"Temp $= 0.1$"
    if "0.5" in col_l:
        return r"Temp $= 0.5$"
    if "5.0" in col_l or "5" in col_l and "temp" in col_l:
        return r"Temp $= 5.0$"
    return col


def _clean_batch_label(col):
    for b in (512, 256, 128, 64, 32):
        if str(b) in col:
            return f"Batch = {b}"
    return col


def _clean_aug_arch_label(col):
    col_l = col.lower()
    if "crop" in col_l and "only" in col_l:
        return "Crop Only"
    if "color" in col_l and "only" in col_l:
        return "Color Only"
    if "no" in col_l and ("projector" in col_l or "proj" in col_l):
        return "No Projector"
    if "baseline" in col_l or "full" in col_l or "0.5" in col_l: # Fallback for baseline run
        return "Baseline (Full)"
    return col.replace("_", " ").replace("-", " ")


def plot_baseline(loss_csv, knn_csv, out_filename="figure1_baseline.pdf"):
    df_loss, x_loss = load_and_clean_csv(loss_csv)
    df_knn, x_knn = load_and_clean_csv(knn_csv)

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax2 = ax1.twinx()

    # Extract default scienceplots colors!
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_loss = colors[0] # First default color
    color_knn = colors[1]  # Second default color

    # Plot loss
    loss_cols = [c for c in df_loss.columns if c != x_loss]
    for c in loss_cols:
        ax1.plot(df_loss[x_loss], df_loss[c], label="NT-Xent Loss", color=color_loss, linewidth=2)
    
    # Match labels/ticks to line color
    ax1.set_ylabel("NT-Xent Loss", color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_xlabel("Epoch")

    # Plot kNN
    knn_cols =[c for c in df_knn.columns if c != x_knn]
    for c in knn_cols:
        ax2.plot(df_knn[x_knn], df_knn[c], label=r"kNN Accuracy (\%)", color=color_knn, linewidth=2)
    
    # Fixed the LaTeX percentage bug here!
    ax2.set_ylabel(r"kNN Accuracy (\%)", color=color_knn)
    ax2.tick_params(axis='y', labelcolor=color_knn)

    # Combine Legends
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig.tight_layout()
    fig.savefig(out_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_ablation(loss_csv, knn_csv, label_cleaner, out_filename, legend_inside=False):
    df_loss, x_loss = load_and_clean_csv(loss_csv)
    df_knn, x_knn = load_and_clean_csv(knn_csv)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Extract default colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Ensure colors match across subplots by mapping labels to colors
    loss_cols =[c for c in df_loss.columns if c != x_loss]
    knn_cols =[c for c in df_knn.columns if c != x_knn]
    all_clean_labels = sorted(list(set([label_cleaner(c) for c in loss_cols + knn_cols])))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(all_clean_labels)}

    # Left: Loss
    ax = axes[0]
    for c in loss_cols:
        lbl = label_cleaner(c)
        ax.plot(df_loss[x_loss], df_loss[c], label=lbl, color=color_map[lbl], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss")
    
    # Right: kNN
    ax = axes[1]
    for c in knn_cols:
        lbl = label_cleaner(c)
        ax.plot(df_knn[x_knn], df_knn[c], label=lbl, color=color_map[lbl], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"kNN Accuracy (\%)") # Fixed percentage bug!

    # Handle legends
    if legend_inside:
        # Put legend on the LEFT graph (axes[0]) at the TOP RIGHT
        axes[0].legend(loc='upper right', fontsize=8)
    else:
        # Place one shared legend below the subplots
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(labels))

    fig.tight_layout()
    # ADDED dpi=300 for high-resolution Word insertion
    fig.savefig(out_filename, bbox_inches='tight', dpi=300) 
    plt.close(fig)


def _clean_eval_label(col):
    col_l = col.lower()
    if "supervised" in col_l:
        return "Supervised Learning"
    if "random_init" in col_l or "random" in col_l:
        return "Random Init"
    if "linear_probe_projector" in col_l or ("linear_probe" in col_l and "projector" in col_l):
        return "Linear Probe (Projector z)"
    if "linear_probe" in col_l:
        return "Linear Probe (Backbone h)"
    return col


def plot_evaluation(train_csv, test_csv, out_filename):
    df_train, x_train = load_and_clean_csv(train_csv)
    df_test, x_test = load_and_clean_csv(test_csv)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Extract default colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Ensure colors match across subplots by mapping labels to colors
    train_cols = [c for c in df_train.columns if c != x_train]
    test_cols = [c for c in df_test.columns if c != x_test]
    all_clean_labels = sorted(list(set([_clean_eval_label(c) for c in train_cols + test_cols])))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(all_clean_labels)}

    # Left: Train Accuracy
    ax = axes[0]
    for c in train_cols:
        lbl = _clean_eval_label(c)
        ax.plot(df_train[x_train], df_train[c], label=lbl, color=color_map[lbl], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Train Accuracy (\%)")

    # Right: Test Accuracy
    ax = axes[1]
    for c in test_cols:
        lbl = _clean_eval_label(c)
        ax.plot(df_test[x_test], df_test[c], label=lbl, color=color_map[lbl], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Test Accuracy (\%)")

    # Shared legend below subplots
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)

    fig.tight_layout()
    fig.savefig(out_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    csv_dir = os.path.join(base_dir, "dataset", "training_data_csv")
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("Generating Figure 1...")
    plot_baseline(
        os.path.join(csv_dir, "baseline_simclr_train_nt_xent_loss.csv"),
        os.path.join(csv_dir, "baseline_simclr_knn_accuracy.csv"),
        out_filename=os.path.join(figures_dir, "figure1_baseline.png"),
    )

    print("Generating Figure 2...")
    plot_ablation(
        os.path.join(csv_dir, "temp_ablation_simclr_train_nt_xent_loss.csv"),
        os.path.join(csv_dir, "temp_ablation_simclr_knn_accuracy.csv"),
        label_cleaner=_clean_temp_label,
        out_filename=os.path.join(figures_dir, "figure2_temp_ablation.png"),
    )

    print("Generating Figure 3...")
    plot_ablation(
        os.path.join(csv_dir, "batch_ablation_simclr_train_nt_xent_loss.csv"),
        os.path.join(csv_dir, "batch_ablation_simclr_knn_accuracy.csv"),
        label_cleaner=_clean_batch_label,
        out_filename=os.path.join(figures_dir, "figure3_batch_ablation.png"),
    )

    print("Generating Figure 4...")
    plot_ablation(
        os.path.join(csv_dir, "augmentation_architecture_ablation_simclr_train_nt_xent_loss.csv"),
        os.path.join(csv_dir, "augmentation_architecture_ablation_simclr_knn_accuracy.csv"),
        label_cleaner=_clean_aug_arch_label,
        out_filename=os.path.join(figures_dir, "figure4_aug_arch_ablation.png"),
        legend_inside=False,
    )

    print("Generating Figure 5 (CIFAR-10)...")
    plot_evaluation(
        os.path.join(csv_dir, "evaluation_train_accuracy_cifar10.csv"),
        os.path.join(csv_dir, "evaluation_test_accuracy_cifar10.csv"),
        out_filename=os.path.join(figures_dir, "figure5_evaluation_cifar10.png"),
    )

    print("Generating Figure 6 (CIFAR-100)...")
    plot_evaluation(
        os.path.join(csv_dir, "evaluation_train_accuracy_cifar100.csv"),
        os.path.join(csv_dir, "evaluation_test_accuracy_cifar100.csv"),
        out_filename=os.path.join(figures_dir, "figure6_evaluation_cifar100.png"),
    )

    print("All plots generated in /figures/ folder at 300 DPI!")
