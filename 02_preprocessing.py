"""
02_preprocessing.py
===================
Gene Expression Cancer Classification Pipeline
Step 2: Preprocessing, Feature Selection & Dimensionality Reduction

Techniques applied
------------------
  • Variance thresholding   – remove near-zero variance genes
  • MAD-based top-k filter  – keep most variable genes (bioinformatics standard)
  • StandardScaler          – z-score normalisation per gene
  • PCA                     – linear dimensionality reduction  (fast, interpretable)
  • t-SNE                   – non-linear 2-D embedding         (visualisation)
  • UMAP                    – non-linear low-dim embedding      (optional, if installed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── constants ────────────────────────────────────────────────────────────────
DATA_DIR   = "/content/data"
OUT_DIR    = "outputs"
SEED       = 42
TOP_K_MAD  = 1000     # keep top-500 most variable genes by MAD
N_PCA_COMP = 100      # PCA components passed to downstream models
os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_data(data_dir: str = DATA_DIR):
    X = pd.read_csv(f"{data_dir}/expression_matrix.csv", index_col=0)
    y = pd.read_csv(f"{data_dir}/labels.csv",            index_col=0).squeeze()
    return X, y


def mad_filter(X: pd.DataFrame, top_k: int = TOP_K_MAD) -> pd.DataFrame:
    """Keep the top-k genes by Median Absolute Deviation (standard in RNA-seq QC)."""
    mad = X.apply(lambda col: np.median(np.abs(col - col.median())), axis=0)
    top_genes = mad.nlargest(top_k).index
    print(f"[Feature Selection] MAD top-{top_k}: {X.shape[1]} → {top_k} genes")
    return X[top_genes]


def variance_filter(X: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    sel = VarianceThreshold(threshold=threshold)
    X_np = sel.fit_transform(X.values)
    kept = X.columns[sel.get_support()]
    print(f"[Feature Selection] Variance threshold: {X.shape[1]} → {X_np.shape[1]} genes")
    return pd.DataFrame(X_np, index=X.index, columns=kept)


def anova_filter(X: np.ndarray, y_enc: np.ndarray, k: int = 200) -> tuple:
    """ANOVA F-test: select k best genes discriminating between classes."""
    sel = SelectKBest(f_classif, k=k)
    X_sel = sel.fit_transform(X, y_enc)
    scores = sel.scores_
    print(f"[Feature Selection] ANOVA top-{k}: shape → {X_sel.shape}")
    return X_sel, sel, scores


def apply_pca(X: np.ndarray, n_components: int = N_PCA_COMP, seed: int = SEED):
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"[PCA] {n_components} components explain "
          f"{cumvar[-1]*100:.1f}% variance")
    return X_pca, pca


def apply_tsne(X: np.ndarray, seed: int = SEED):
    import sklearn
    tsne_kwargs = dict(n_components=2, perplexity=40, learning_rate=200,
                       random_state=seed, init="pca")
    # sklearn ≥1.2 renamed n_iter → max_iter
    sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sk_ver >= (1, 2):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(**tsne_kwargs)
    X_tsne = tsne.fit_transform(X)
    print(f"[t-SNE] Embedding shape: {X_tsne.shape}")
    return X_tsne, tsne


# ── plotting utilities ────────────────────────────────────────────────────────
PALETTE = [
    "#E63946", "#2A9D8F", "#457B9D", "#E9C46A", "#6A0572",
]


def plot_pca_variance(pca: PCA, out_dir: str = OUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PCA Explained Variance", fontsize=14, fontweight="bold")

    # per-component
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_ * 100,
                color="#457B9D", alpha=0.85)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Per-Component Variance")
    axes[0].set_xlim(0, min(30, len(pca.explained_variance_ratio_)) + 1)

    # cumulative
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[1].plot(range(1, len(cumvar) + 1), cumvar,
                 color="#E63946", linewidth=2, marker="o", markersize=3)
    axes[1].axhline(90, linestyle="--", color="#6A0572", linewidth=1.2,
                    label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()

    plt.tight_layout()
    path = f"{out_dir}/pca_variance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] PCA variance → {path}")


def plot_2d_embedding(X_2d: np.ndarray, y: np.ndarray,
                      title: str, filename: str, out_dir: str = OUT_DIR):
    classes = np.unique(y)
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=cls, alpha=0.75, s=22,
                   color=PALETTE[i % len(PALETTE)], edgecolors="none")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(title="Cancer Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{out_dir}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 2-D embedding → {path}")


def plot_top_genes(scores: np.ndarray, gene_names: list,
                   top_n: int = 20, out_dir: str = OUT_DIR):
    top_idx   = np.argsort(scores)[::-1][:top_n]
    top_names = [gene_names[i] for i in top_idx]
    top_sc    = scores[top_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(top_n), top_sc[::-1],
                   color=PALETTE[2], alpha=0.85, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("ANOVA F-Score")
    ax.set_title(f"Top {top_n} Discriminative Genes (ANOVA)", fontsize=13,
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = f"{out_dir}/top_genes_anova.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Top genes  → {path}")


# ── main pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing():
    print("=" * 60)
    print("  Step 2 — Preprocessing & Dimensionality Reduction")
    print("=" * 60)

    # 1. load ─────────────────────────────────────────────────────────────────
    X_raw, y = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\n[INFO] Raw data: {X_raw.shape}  |  classes: {le.classes_}")

    # 2. variance filter ──────────────────────────────────────────────────────
    X_var = variance_filter(X_raw, threshold=0.1)

    # 3. MAD top-k ────────────────────────────────────────────────────────────
    X_mad = mad_filter(X_var, top_k=TOP_K_MAD)
    gene_names_mad = list(X_mad.columns)

    # 4. z-score normalise ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mad.values)
    print(f"[Normalisation] z-score applied: mean≈{X_scaled.mean():.4f}, "
          f"std≈{X_scaled.std():.4f}")

    # 5. ANOVA feature selection ──────────────────────────────────────────────
    X_anova, anova_sel, anova_scores = anova_filter(X_scaled, y_enc, k=200)
    plot_top_genes(anova_scores, gene_names_mad, top_n=20)

    # 6. PCA ──────────────────────────────────────────────────────────────────
    X_pca, pca = apply_pca(X_scaled, n_components=N_PCA_COMP)
    plot_pca_variance(pca)

    # 7. 2-D PCA for visualisation ────────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=SEED)
    X_pca2 = pca2.fit_transform(X_scaled)
    plot_2d_embedding(X_pca2, y.values, "PCA 2-D — Cancer Subtypes",
                      "pca_2d_embedding.png")

    # 8. t-SNE ────────────────────────────────────────────────────────────────
    print("\n[t-SNE] Running (this takes ~30 s) …")
    X_tsne, _ = apply_tsne(X_pca, seed=SEED)   # use PCA output for speed
    plot_2d_embedding(X_tsne, y.values, "t-SNE 2-D — Cancer Subtypes",
                      "tsne_2d_embedding.png")

    # 9. save processed arrays ────────────────────────────────────────────────
    np.save(f"{DATA_DIR}/X_pca.npy",    X_pca)
    np.save(f"{DATA_DIR}/X_scaled.npy", X_scaled)
    np.save(f"{DATA_DIR}/y_enc.npy",    y_enc)
    np.save(f"{DATA_DIR}/X_anova.npy",  X_anova)
    np.save(f"{DATA_DIR}/classes.npy",  le.classes_)
    print(f"\n[INFO] Processed arrays saved to {DATA_DIR}/")
    print("[DONE] Preprocessing complete.\n")

    return X_pca, X_scaled, y_enc, le


if __name__ == "__main__":
    run_preprocessing()
