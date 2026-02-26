# Gene Expression–Based Cancer Classification

> Bioinformatics · Scikit-learn · Dimensionality Reduction  

---

## Overview

An end-to-end Machine Learning pipeline that classifies **5 cancer subtypes**
from real TCGA gene expression data using SVM and Random Forest —
achieving **96.89% accuracy** across 801 real patients × 20,531 genes.

---

## Results

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| SVM (RBF) | **96.89%** | 0.9654 | 0.9803 |
| Random Forest | **96.89%** | 0.9654 | 0.9777 |

---

## Cancer Types

| Code | Cancer | Patients |
|---|---|---|
| BRCA | Breast Cancer | 300 |
| KIRC | Kidney Cancer | 146 |
| LUAD | Lung Cancer | 141 |
| PRAD | Prostate Cancer | 136 |
| COAD | Colon Cancer | 78 |
| | **Total** | **801** |

---

## Dataset

- **Source:** TCGA Pan-Cancer — [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq)
- **Samples:** 801 real patient samples
- **Genes:** 20,531 RNA-Seq expression values

---

## Pipeline

```
Step 1 → Data Loading       (801 × 20,531 TCGA matrix)
Step 2 → Preprocessing      (MAD filter, ANOVA, Z-score, PCA, t-SNE)
Step 3 → Classification     (SVM + Random Forest + GridSearchCV)
Step 4 → Biomarker Discovery (69 DE genes, Volcano plots, Heatmap)
```

---
## Visual Results

To evaluate the performance of the cancer classification models, we analyzed both ROC curves and confusion matrices generated from the test dataset.

**Confusion Matrix Analysis**
The confusion matrix illustrates how many samples from each cancer type were correctly classified versus misclassified. The model shows strong diagonal dominance, indicating high prediction accuracy across major cancer classes. Minor misclassifications appear primarily in classes with fewer samples, reflecting dataset imbalance rather than model instability.
<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/aab35304-4c91-4cf8-bf73-9204a5274614" />



**ROC Curve Interpretation**
The multi-class ROC curves measure the model’s ability to distinguish between different cancer subtypes. Both SVM and Random Forest achieved near-ideal ROC behavior, with macro AUC values close to 0.98. This indicates that the models learned highly discriminative patterns in gene expression space and can reliably separate cancer classes beyond random chance.
<img width="600" height="500" alt="roc_curve" src="https://github.com/user-attachments/assets/df8fcfab-f16f-43a0-b0e7-8a7bb623f6b0" />


**Overall Insight**
Together, the confusion matrix and ROC curves confirm that gene expression signatures provide strong predictive power for cancer subtype identification. The evaluation demonstrates that the pipeline generalizes well to unseen patients while maintaining high sensitivity and specificity.


## Key Techniques

- Variance threshold + MAD filtering: 20,531 → 1,000 genes
- ANOVA F-test feature selection: → 200 genes
- PCA: 100 components, **83.2% variance explained**
- t-SNE: 2D cluster visualisation
- GridSearchCV hyperparameter tuning (5-fold CV)
- 69 differentially expressed biomarker genes identified

---

## Visualisations

| Plot | Description |
|---|---|
| `tsne_2d_embedding.png` | 2D patient cluster map |
| `heatmap_top_genes.png` | Top DE genes across samples |
| `model_comparison.png` | SVM vs Random Forest |
| `volcano_BRCA.png` | Biomarkers in Breast Cancer |
| `svm_confusion_matrix.png` | Per-class accuracy |

---

## Tech Stack

```
Python · Scikit-learn · Pandas · NumPy · Matplotlib · Google Colab
```

---

## Project Structure

```
gene_expression_cancer_classification/
│
├── gene_expression_cancer_classification.ipynb
├── 02_preprocessing.py
├── 03_classification.py
├── 04_biomarker_visualization.py
├── README.md
└── outputs/
    ├── tsne_2d_embedding.png
    ├── heatmap_top_genes.png
    ├── model_comparison.png
    └── svm_confusion_matrix.png
```

---

## How to Run

```bash
# 1. Clone repo
git clone https://github.com/yourusername/gene_expression_cancer_classification

# 2. Install dependencies
pip install numpy pandas matplotlib scikit-learn scipy joblib

# 3. Open notebook in Google Colab
gene_expression_cancer_classification.ipynb
```

---

*Real TCGA data · 801 patients · 20,531 genes · 96.89% accuracy*
