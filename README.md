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

## Tested algorithm cores

The ML behind the demos lives in framework-free ES modules under [`src/`](src/) — no DOM, no
dependencies. `index.html` loads `src/app.js`, which imports these modules for every demo and is
responsible only for reading controls and rendering. The same modules are unit-tested with Node's
built-in test runner.

| Module | Algorithm | Tested properties |
| --- | --- | --- |
| `src/rng.js` | seedable `mulberry32` + Box–Muller `gauss` | reproducible streams (used by every other test) |
| `src/split.js` | train/test split (Fisher–Yates) | true partition: no overlap, full coverage, `round(n·f)` sizes, seed-reproducible |
| `src/linreg.js` | linear regression by gradient descent; least-squares `polyfit` | recovers a known slope/intercept on noiseless data; loss decreases every step; exact polynomial recovery |
| `src/logreg.js` | logistic regression, GD on cross-entropy | separates a separable set (≥95% train acc); cross-entropy decreases monotonically |
| `src/knn.js` | k-NN classify + leave-one-out accuracy | classifies separable clusters; votes sum to k; LOO ≥95% |
| `src/kmeans.js` | Lloyd's algorithm + inertia (WCSS) | converges to known blobs; inertia non-increasing per iteration |
| `src/tree.js` | Gini impurity + greedy best split | finds the clean boundary; perfectly separates a separable set |
| `src/scaling.js` | z-score standardisation + min–max | standardised axes have zero mean / unit variance; min–max maps to [0,1] |
| `src/metrics.js` | confusion matrix, accuracy/precision/recall/F₁, MSE, R², ROC/AUC | match hand-computed values on constructed cases |
| `src/pca.js` | 2×2 covariance + symmetric eigen-decomposition | eigenvector check `A v = λ v`; PC1 aligns with the dominant direction |

## Run it

It's a static site — open `index.html`, or serve the folder and visit it:

```sh
npx http-server . -p 8000   # then open http://localhost:8000
```

## Tests

No dependencies. Requires Node 18+ (developed on Node 24, `node:test` + `node:assert`):

```sh
node --test
```

Real output:

```
ℹ tests 45
ℹ suites 0
ℹ pass 45
ℹ fail 0
ℹ cancelled 0
ℹ skipped 0
ℹ todo 0
ℹ duration_ms 311.8869
```

Part of the *-lab series: [modeling-lab](https://github.com/andreaisabelmontana/modeling-lab) · [stats-lab](https://github.com/andreaisabelmontana/stats-lab) · [linalg-lab](https://github.com/andreaisabelmontana/linalg-lab) · [data-analysis-lab](https://github.com/andreaisabelmontana/data-analysis-lab)
