# ml-lab

Visual machine-learning foundations. 11 self-contained demos walking the core ML pipeline — from data preparation through training, evaluation and the classic algorithms — for the IE BCSAI *AI: Machine Learning Foundations* course (Burkov, *Machine Learning Engineering*).

**Live:** https://andreaisabelmontana.github.io/ml-lab/

The pipeline & generalization
1. Train/test split & overfitting — fit a polynomial, watch test RMSE climb as degree grows while train RMSE keeps falling
2. Gradient descent — minimise MSE for linear regression step by step; large learning rates diverge
3. Bias–variance tradeoff — the bias²/variance/noise decomposition and the complexity that minimises total error

Data preparation
4. Feature scaling & standardisation — z-score and min–max transforms of a 2-D cloud, with live mean/std/range

Supervised algorithms
5. k-nearest neighbours — clickable decision regions and majority-vote classification, with leave-one-out accuracy
6. Logistic regression — a perceptron-style linear classifier trained by gradient descent on cross-entropy
7. Decision tree — greedy Gini-impurity splits carving the plane into axis-aligned leaf boxes

Unsupervised algorithms
8. k-means clustering — Lloyd's assign/update loop animated, with falling inertia (WCSS)
9. Principal component analysis — covariance eigen-decomposition; project onto PC1 and read variance retained

Model evaluation
10. Confusion matrix & threshold — slide the decision threshold; read accuracy, precision, recall and F₁ live
11. ROC curve & AUC — sweep the threshold to trace TPR vs FPR; AUC measures ranking quality

Plain HTML + canvas + KaTeX. Indigo accent. Zero build step.

Part of the *-lab series: [modeling-lab](https://github.com/andreaisabelmontana/modeling-lab) · [stats-lab](https://github.com/andreaisabelmontana/stats-lab) · [linalg-lab](https://github.com/andreaisabelmontana/linalg-lab) · [data-analysis-lab](https://github.com/andreaisabelmontana/data-analysis-lab)
