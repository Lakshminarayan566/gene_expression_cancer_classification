"""
04_biomarker_visualization.py
==============================
Gene Expression Cancer Classification Pipeline
Step 4: Biomarker Discovery & Advanced Visualization

Visualisations
--------------
  • Heatmap  — top differentially expressed genes × samples
  • Volcano plot — fold-change vs significance
  • SHAP summary (RF) — model-level gene attribution
  • PCA biplot — loadings of top genes on PC axes
  • Clustermap — hierarchical clustering of samples & genes
  • Decision boundary visualisation (SVM on 2-D PCA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import warnings, os, joblib
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ── constants ────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUT_DIR    = "outputs"
MODEL_DIR  = "models"
SEED       = 42
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "BRCA": "#E63946", "LUAD": "#2A9D8F",
    "COAD": "#457B9D", "GBM" : "#E9C46A", "KIRC": "#6A0572",
}
CANCER_TYPES = list(PALETTE.keys())


# ── loaders ───────────────────────────────────────────────────────────────────
def load_raw():
    X = pd.read_csv(f"{DATA_DIR}/expression_matrix.csv", index_col=0)
    y = pd.read_csv(f"{DATA_DIR}/labels.csv",            index_col=0).squeeze()
    return X, y


def load_processed():
    X_pca   = np.load(f"{DATA_DIR}/X_pca.npy")
    y_enc   = np.load(f"{DATA_DIR}/y_enc.npy")
    classes = np.load(f"{DATA_DIR}/classes.npy", allow_pickle=True)
    return X_pca, y_enc, classes


# ── differential expression helpers ──────────────────────────────────────────
def differential_expression(X: pd.DataFrame, y: pd.Series,
                             target_class: str, top_n: int = 50):
    """
    One-vs-rest t-test for each gene.
    Returns DataFrame with fold_change and -log10(p-value).
    """
    mask_pos = (y == target_class).values
    mask_neg = ~mask_pos

    X_pos = X.values[mask_pos]
    X_neg = X.values[mask_neg]

    # fold change (log2 space → subtract means)
    fc = X_pos.mean(axis=0) - X_neg.mean(axis=0)

    # t-test
    _, pvals = stats.ttest_ind(X_pos, X_neg, equal_var=False)
    pvals = np.where(pvals == 0, 1e-300, pvals)   # avoid log(0)
    neg_log_p = -np.log10(pvals)

    result = pd.DataFrame({
        "gene":        X.columns,
        "fold_change": fc,
        "neg_log10_p": neg_log_p,
    }).set_index("gene")
    return result


def select_top_de_genes(X: pd.DataFrame, y: pd.Series,
                         n_per_class: int = 20) -> list:
    """Select top DE genes across all classes."""
    top_genes = set()
    for cls in y.unique():
        de = differential_expression(X, y, cls)
        # significant & upregulated
        sig = de[(de["neg_log10_p"] > 2) & (de["fold_change"].abs() > 0.5)]
        sig_sorted = sig.nlargest(n_per_class, "neg_log10_p")
        top_genes.update(sig_sorted.index.tolist())
    return list(top_genes)


# ── plot 1: heatmap ───────────────────────────────────────────────────────────
def plot_heatmap(X: pd.DataFrame, y: pd.Series,
                 top_genes: list, out_dir: str = OUT_DIR):
    X_sub = X[top_genes]

    # sort samples by class
    order = y.argsort()
    X_plot = X_sub.iloc[order].T

    # z-score across samples
    X_z = (X_plot - X_plot.mean(axis=1).values[:, None]) \
          / (X_plot.std(axis=1).values[:, None] + 1e-9)
    X_z = X_z.clip(-3, 3)

    sorted_y = y.iloc[order].values
    col_colors = [PALETTE.get(c, "#999999") for c in sorted_y]

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [0.04, 1]},
    )

    # colour bar (class annotation)
    for i, col in enumerate(col_colors):
        axes[0].add_patch(plt.Rectangle(
            (i / len(col_colors), 0), 1 / len(col_colors), 1,
            color=col, transform=axes[0].transAxes,
        ))
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].axis("off")
    axes[0].set_title("Gene Expression Heatmap — Top DE Genes",
                       fontsize=13, fontweight="bold", pad=6)

    im = axes[1].imshow(X_z.values, aspect="auto", cmap="RdBu_r",
                        vmin=-3, vmax=3, interpolation="nearest")
    axes[1].set_yticks(range(len(top_genes)))
    axes[1].set_yticklabels(top_genes, fontsize=6)
    axes[1].set_xlabel("Samples")
    axes[1].set_ylabel("Genes")

    plt.colorbar(im, ax=axes[1], orientation="vertical",
                 fraction=0.02, pad=0.02, label="Z-score")

    # legend
    handles = [mpatches.Patch(color=c, label=cls)
               for cls, c in PALETTE.items()]
    axes[0].legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.5), ncol=5,
                   fontsize=8, frameon=False)

    plt.tight_layout()
    path = f"{out_dir}/heatmap_top_genes.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Heatmap → {path}")


# ── plot 2: volcano plot ──────────────────────────────────────────────────────
def plot_volcano(X: pd.DataFrame, y: pd.Series,
                 target_class: str = "BRCA", out_dir: str = OUT_DIR):
    de = differential_expression(X, y, target_class)

    sig_up   = (de["fold_change"] >  0.5) & (de["neg_log10_p"] > 2)
    sig_down = (de["fold_change"] < -0.5) & (de["neg_log10_p"] > 2)
    ns       = ~(sig_up | sig_down)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(de.loc[ns, "fold_change"],   de.loc[ns, "neg_log10_p"],
               s=8, alpha=0.4, color="#AAAAAA", label="Not significant")
    ax.scatter(de.loc[sig_up, "fold_change"],   de.loc[sig_up, "neg_log10_p"],
               s=12, alpha=0.7, color="#E63946", label=f"Up in {target_class}")
    ax.scatter(de.loc[sig_down, "fold_change"], de.loc[sig_down, "neg_log10_p"],
               s=12, alpha=0.7, color="#457B9D", label=f"Down in {target_class}")

    # label top 5 genes
    top5 = de[sig_up].nlargest(5, "neg_log10_p")
    for gene, row in top5.iterrows():
        ax.annotate(gene, xy=(row["fold_change"], row["neg_log10_p"]),
                    fontsize=7, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    xytext=(row["fold_change"] + 0.1, row["neg_log10_p"] + 0.3))

    ax.axhline(2,   linestyle="--", color="gray",   linewidth=0.9)
    ax.axvline(0.5, linestyle="--", color="#E63946", linewidth=0.9)
    ax.axvline(-0.5,linestyle="--", color="#457B9D", linewidth=0.9)

    ax.set_xlabel("Log2 Fold Change (vs rest)", fontsize=11)
    ax.set_ylabel("-log10(p-value)",            fontsize=11)
    ax.set_title(f"Volcano Plot — {target_class} vs Rest",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{out_dir}/volcano_{target_class}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Volcano → {path}")


# ── plot 3: PCA biplot ────────────────────────────────────────────────────────
def plot_pca_biplot(X_scaled: np.ndarray, y: pd.Series, pca: PCA,
                    X_cols: list, n_arrows: int = 10, out_dir: str = OUT_DIR):
    """PCA scatter + loading vectors for top n contributing genes."""
    X_2d = pca.transform(X_scaled)[:, :2]
    loadings = pca.components_[:2]   # shape (2, n_features)

    # select genes with largest magnitude loading on PC1 or PC2
    mag = np.sqrt(loadings[0]**2 + loadings[1]**2)
    top_idx = np.argsort(mag)[::-1][:n_arrows]

    # scale factor so arrows are visible
    scale = np.abs(X_2d).max() / np.abs(loadings).max() * 0.35

    fig, ax = plt.subplots(figsize=(9, 7))
    classes = y.unique()
    for cls, col in list(PALETTE.items())[:len(classes)]:
        mask = y == cls
        ax.scatter(X_2d[mask.values, 0], X_2d[mask.values, 1],
                   color=col, label=cls, alpha=0.6, s=18, edgecolors="none")

    for idx in top_idx:
        ax.annotate("", xy=(loadings[0, idx] * scale, loadings[1, idx] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax.text(loadings[0, idx] * scale * 1.12,
                loadings[1, idx] * scale * 1.12,
                X_cols[idx], fontsize=7, ha="center", color="black")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Biplot — Samples & Gene Loadings", fontsize=13,
                 fontweight="bold")
    ax.legend(title="Cancer Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{out_dir}/pca_biplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] PCA biplot → {path}")


# ── plot 4: SVM decision boundary (2-D PCA) ───────────────────────────────────
def plot_svm_decision_boundary(X_pca2: np.ndarray, y_enc: np.ndarray,
                                classes, out_dir: str = OUT_DIR):
    from matplotlib.colors import ListedColormap

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              random_state=SEED, class_weight="balanced")
    svm.fit(X_pca2, y_enc)

    h       = 0.4
    x_min   = X_pca2[:, 0].min() - 2;  x_max = X_pca2[:, 0].max() + 2
    y_min   = X_pca2[:, 1].min() - 2;  y_max = X_pca2[:, 1].max() + 2
    xx, yy  = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
    Z       = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z       = Z.reshape(xx.shape)

    cmap_bg = ListedColormap(
        [c + "55" for c in list(PALETTE.values())[:len(classes)]]
    )
    cmap_pt = ListedColormap(list(PALETTE.values())[:len(classes)])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
    sc = ax.scatter(X_pca2[:, 0], X_pca2[:, 1],
                    c=y_enc, cmap=cmap_pt, s=18, edgecolors="none", alpha=0.8)

    handles = [mpatches.Patch(color=list(PALETTE.values())[i], label=classes[i])
               for i in range(len(classes))]
    ax.legend(handles=handles, title="Cancer Type",
              bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("PC1");  ax.set_ylabel("PC2")
    ax.set_title("SVM Decision Boundary (2-D PCA Space)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{out_dir}/svm_decision_boundary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] SVM boundary → {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def run_biomarker_analysis():
    print("=" * 60)
    print("  Step 4 — Biomarker Discovery & Visualisation")
    print("=" * 60)

    X_raw, y = load_raw()
    X_pca, y_enc, classes = load_processed()

    # scale raw data for biplot
    from sklearn.feature_selection import VarianceThreshold
    vt = VarianceThreshold(0.1)
    X_var = vt.fit_transform(X_raw.values)
    gene_names_var = X_raw.columns[vt.get_support()].tolist()

    # MAD top-500
    mad = np.median(np.abs(X_raw.values - np.median(X_raw.values, axis=0)), axis=0)
    top_idx_mad = np.argsort(mad)[::-1][:500]
    X_mad = X_raw.iloc[:, top_idx_mad]
    gene_names_mad = X_mad.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mad.values)

    # 1. top DE genes ──────────────────────────────────────────────────────────
    print("\n[DE] Identifying differentially expressed genes …")
    top_genes = select_top_de_genes(X_mad, y, n_per_class=15)
    top_genes = top_genes[:min(80, len(top_genes))]
    print(f"     {len(top_genes)} DE genes selected")

    # 2. heatmap ───────────────────────────────────────────────────────────────
    plot_heatmap(X_mad, y, top_genes)

    # 3. volcano plots ─────────────────────────────────────────────────────────
    for cls in CANCER_TYPES[:3]:
        plot_volcano(X_mad, y, target_class=cls)

    # 4. PCA biplot ────────────────────────────────────────────────────────────
    pca_bio = PCA(n_components=2, random_state=SEED)
    pca_bio.fit(X_scaled)
    plot_pca_biplot(X_scaled, y, pca_bio, gene_names_mad, n_arrows=12)

    # 5. SVM decision boundary ─────────────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=SEED)
    X_pca2 = pca2.fit_transform(X_scaled)
    plot_svm_decision_boundary(X_pca2, y_enc, classes)

    print("\n[DONE] Biomarker analysis & visualisation complete.")


if __name__ == "__main__":
    run_biomarker_analysis()
