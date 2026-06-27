// Classification and regression metrics.
//
// Classification metrics take parallel arrays of binary truth and prediction
// labels (0/1). `confusionMatrix` returns the four counts; accuracy, precision,
// recall and F1 are derived from them. `confusionFromScores` thresholds a list
// of scored examples first. Regression metrics (`mse`, `r2`) take parallel
// arrays of true and predicted values.

// { tp, fp, fn, tn } for parallel 0/1 arrays.
export function confusionMatrix(yTrue, yPred) {
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const y = yTrue[i],
      p = yPred[i];
    if (y === 1 && p === 1) tp++;
    else if (y === 0 && p === 1) fp++;
    else if (y === 1 && p === 0) fn++;
    else tn++;
  }
  return { tp, fp, fn, tn };
}

// Threshold scored examples [{s, y}] at `thr` (predict 1 iff s >= thr), then
// return the confusion matrix.
export function confusionFromScores(examples, thr) {
  const yTrue = examples.map((e) => e.y);
  const yPred = examples.map((e) => (e.s >= thr ? 1 : 0));
  return confusionMatrix(yTrue, yPred);
}

export function accuracy(cm) {
  const tot = cm.tp + cm.fp + cm.fn + cm.tn;
  return tot ? (cm.tp + cm.tn) / tot : 0;
}

export function precision(cm) {
  return cm.tp + cm.fp ? cm.tp / (cm.tp + cm.fp) : 0;
}

export function recall(cm) {
  return cm.tp + cm.fn ? cm.tp / (cm.tp + cm.fn) : 0;
}

export function f1(cm) {
  const p = precision(cm),
    r = recall(cm);
  return p + r ? (2 * p * r) / (p + r) : 0;
}

// Mean squared error between parallel arrays of true and predicted values.
export function mse(yTrue, yPred) {
  const n = yTrue.length;
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += (yTrue[i] - yPred[i]) ** 2;
  return s / n;
}

// Coefficient of determination R^2 = 1 - SS_res / SS_tot.
export function r2(yTrue, yPred) {
  const n = yTrue.length;
  if (!n) return 0;
  const mean = yTrue.reduce((a, v) => a + v, 0) / n;
  let ssRes = 0,
    ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (yTrue[i] - yPred[i]) ** 2;
    ssTot += (yTrue[i] - mean) ** 2;
  }
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}

// Sweep all thresholds to trace the ROC curve and integrate AUC by the
// trapezoidal rule. `pos` / `neg` are arrays of scores for the two classes.
export function rocAuc(pos, neg) {
  const rates = (thr) => {
    const tp = pos.filter((s) => s >= thr).length;
    const fn = pos.length - tp;
    const fp = neg.filter((s) => s >= thr).length;
    const tn = neg.length - fp;
    return { tpr: tp / (tp + fn || 1), fpr: fp / (fp + tn || 1) };
  };
  const pts = [];
  for (let t = 1.001; t >= -0.001; t -= 0.01) {
    const { tpr, fpr } = rates(t);
    pts.push({ fpr, tpr });
  }
  pts.sort((a, b) => a.fpr - b.fpr || a.tpr - b.tpr);
  let auc = 0;
  for (let i = 1; i < pts.length; i++) {
    auc += ((pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr)) / 2;
  }
  return { pts, auc };
}
