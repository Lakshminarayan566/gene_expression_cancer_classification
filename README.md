# 🧬 Gene Expression–Based Cancer Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Accuracy-96.89%25-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Dataset-TCGA%20Pan--Cancer-purple?style=flat-square" />
</p>

> An end-to-end ML pipeline that classifies **5 cancer subtypes** from real TCGA RNA-Seq data using SVM and Random Forest — achieving **96.89% accuracy** across 801 patients × 20,531 genes.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Visual Results](#-visual-results)
- [Key Techniques](#-key-techniques)
- [Visualisations](#-visualisations)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)

---

## 🔬 Overview

Cancer subtype identification from high-dimensional gene expression profiles is a critical challenge in precision oncology. This project builds a complete bioinformatics ML pipeline on the **TCGA Pan-Cancer RNA-Seq dataset**, taking raw expression data through aggressive dimensionality reduction and into two high-performing classifiers.

The pipeline goes from **20,531 raw gene features → 69 key biomarker genes**, ultimately achieving near-perfect discrimination across 5 clinically distinct cancer types — all verified on real patient data.

---

## 📊 Results

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| SVM (RBF) | **96.89%** | 0.9654 | 0.9803 |
| Random Forest | **96.89%** | 0.9654 | 0.9777 |

> Both models trained with 5-fold GridSearchCV cross-validation on 801 real TCGA patient samples.

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| **Source** | TCGA Pan-Cancer — [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq) |
| **Samples** | 801 real patient RNA-Seq profiles |
| **Features** | 20,531 gene expression values |
| **Classes** | 5 cancer subtypes |

### Cancer Subtypes

| Code | Cancer | Patients |
|---|---|---|
| BRCA | Breast Cancer | 300 |
| KIRC | Kidney (Renal) Cancer | 146 |
| LUAD | Lung Adenocarcinoma | 141 |
| PRAD | Prostate Cancer | 136 |
| COAD | Colon Adenocarcinoma | 78 |
| | **Total** | **801** |

---

## ⚙️ Pipeline

```
Step 1 → Data Loading        (801 × 20,531 TCGA matrix)
Step 2 → Preprocessing       (MAD filter → ANOVA → Z-score → PCA → t-SNE)
Step 3 → Classification      (SVM + Random Forest + GridSearchCV)
Step 4 → Biomarker Discovery (69 DE genes · Volcano plots · Heatmap)
```

### Dimensionality Reduction at a Glance

```
20,531 genes
   └─ MAD Variance Filter      → 1,000 genes
       └─ ANOVA F-test          →   200 genes
           └─ PCA (100 comps)   →  83.2% variance explained
               └─ t-SNE (2D)    →  Visual cluster separation
```

---

## 📈 Visual Results

### Confusion Matrix

The matrix shows strong diagonal dominance across all 5 cancer types. Minor off-diagonal misclassifications occur mainly in COAD (n=78), the smallest class — a reflection of class imbalance rather than model instability.

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/aab35304-4c91-4cf8-bf73-9204a5274614" />

### ROC Curves

Multi-class ROC curves for both classifiers show near-ideal behaviour, with macro AUC values of **0.9803 (SVM)** and **0.9777 (RF)**. This confirms that gene expression signatures form highly discriminative boundaries between cancer subtypes in reduced feature space.

<img width="600" height="500" alt="roc_curve" src="https://github.com/user-attachments/assets/df8fcfab-f16f-43a0-b0e7-8a7bb623f6b0" />

---

## 🧪 Key Techniques

| Technique | Detail |
|---|---|
| MAD Variance Filtering | 20,531 → 1,000 genes |
| ANOVA F-test Selection | 1,000 → 200 genes |
| PCA | 100 components · 83.2% variance explained |
| t-SNE | 2D cluster visualisation |
| SVM (RBF Kernel) | GridSearchCV tuned · 96.89% accuracy |
| Random Forest | GridSearchCV tuned · 96.89% accuracy |
| Biomarker Discovery | 69 differentially expressed genes identified |

---

## 🖼️ Visualisations

| File | Description |
|---|---|
| `tsne_2d_embedding.png` | 2D patient cluster map by cancer subtype |
| `heatmap_top_genes.png` | Expression heatmap of top DE genes |
| `model_comparison.png` | SVM vs Random Forest performance comparison |
| `volcano_BRCA.png` | Volcano plot — BRCA biomarker genes |
| `svm_confusion_matrix.png` | Per-class prediction accuracy |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Scikit-learn | SVM, Random Forest, GridSearchCV, PCA |
| Pandas / NumPy | Data wrangling |
| Matplotlib / Seaborn | Visualisation |
| SciPy | ANOVA F-test, statistical analysis |
| Google Colab | Development environment |

---

## 📁 Project Structure

```
gene_expression_cancer_classification/
│
├── gene_expression_cancer_classification.ipynb   ← Main notebook
├── 02_preprocessing.py                           ← Feature filtering & reduction
├── 03_classification.py                          ← SVM & RF training
├── 04_biomarker_visualization.py                 ← DE gene analysis & plots
├── README.md
│
└── outputs/
    ├── tsne_2d_embedding.png
    ├── heatmap_top_genes.png
    ├── model_comparison.png
    └── svm_confusion_matrix.png
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Lakshminarayan566/gene_expression_cancer_classification
cd gene_expression_cancer_classification

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib

# 3. Download the dataset
# → https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq
# Place data.csv and labels.csv in the project root

# 4. Run the notebook
# → Open gene_expression_cancer_classification.ipynb in Google Colab or Jupyter
```

> **Note:** The dataset is not included in this repo due to size. Download it directly from the UCI ML Repository link above.

---

<p align="center">
  Real TCGA Data · 801 Patients · 20,531 Genes · 96.89% Accuracy
</p>
