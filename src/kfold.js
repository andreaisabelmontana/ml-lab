// k-fold cross-validation — a less optimistic, lower-variance estimate of
// generalization than a single train/test split (Model Evaluation, session 13).
// The data is shuffled and split into k folds; each fold serves once as the
// validation set while the other k-1 train the model, and the validation
// errors are averaged. Pure and DOM-free; builds on the tested linreg core.

import { mulberry32 } from './rng.js';
import { polyfit, evalPoly, rmse } from './linreg.js';

// Partition indices 0..n-1 into k folds of nearly equal size after a seeded
// shuffle. Returns an array of k arrays of indices.
export function kFoldPartition(n, k, seed = 1) {
  const idx = Array.from({ length: n }, (_, i) => i);
  const rng = mulberry32(seed);
  for (let i = n - 1; i > 0; i--) {            // Fisher–Yates shuffle
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const folds = Array.from({ length: k }, () => []);
  idx.forEach((v, i) => folds[i % k].push(v));  // round-robin ⇒ balanced sizes
  return folds;
}

// Cross-validate a degree-`degree` polynomial fit over `points` ({x,y}).
// Returns per-fold results (the held-out indices, the fitted coefficients and
// the validation RMSE) plus the mean and standard deviation of the fold RMSEs.
export function crossValidate(points, { degree = 3, k = 5, seed = 1 } = {}) {
  const n = points.length;
  const kk = Math.max(2, Math.min(k, n));
  const folds = kFoldPartition(n, kk, seed);
  const results = folds.map((valIdx) => {
    const valSet = new Set(valIdx);
    const train = points.filter((_, i) => !valSet.has(i));
    const val = valIdx.map(i => points[i]);
    const coef = polyfit(train, degree);
    return { valIdx, coef, rmse: rmse(val, coef) };
  });
  const errs = results.map(r => r.rmse);
  const mean = errs.reduce((s, e) => s + e, 0) / errs.length;
  const variance = errs.reduce((s, e) => s + (e - mean) ** 2, 0) / errs.length;
  return { results, mean, std: Math.sqrt(variance), k: kk };
}

// Convenience: the cross-validated RMSE for a polynomial of each degree in
// `degrees`, useful for picking the degree that generalizes best.
export function cvCurve(points, degrees, { k = 5, seed = 1 } = {}) {
  return degrees.map(degree => ({ degree, cvRmse: crossValidate(points, { degree, k, seed }).mean }));
}

export { evalPoly };
