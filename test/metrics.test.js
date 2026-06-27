import test from 'node:test';
import assert from 'node:assert/strict';
import {
  confusionMatrix,
  confusionFromScores,
  accuracy,
  precision,
  recall,
  f1,
  mse,
  r2,
  rocAuc,
} from '../src/metrics.js';

test('confusion matrix on a constructed case', () => {
  //            pred1 pred0
  // actual1     TP=2  FN=1
  // actual0     FP=1  TN=2
  const yTrue = [1, 1, 1, 0, 0, 0];
  const yPred = [1, 1, 0, 1, 0, 0];
  const cm = confusionMatrix(yTrue, yPred);
  assert.deepEqual(cm, { tp: 2, fp: 1, fn: 1, tn: 2 });
});

test('accuracy / precision / recall / f1 match hand-computed values', () => {
  const cm = { tp: 2, fp: 1, fn: 1, tn: 2 };
  // accuracy = (2+2)/6 = 0.6667
  assert.ok(Math.abs(accuracy(cm) - 4 / 6) < 1e-12);
  // precision = 2/(2+1) = 0.6667
  assert.ok(Math.abs(precision(cm) - 2 / 3) < 1e-12);
  // recall = 2/(2+1) = 0.6667
  assert.ok(Math.abs(recall(cm) - 2 / 3) < 1e-12);
  // f1 = 2*p*r/(p+r) = 2/3
  assert.ok(Math.abs(f1(cm) - 2 / 3) < 1e-12);
});

test('precision/recall on an asymmetric case', () => {
  // tp=3, fp=1, fn=2, tn=4
  const cm = { tp: 3, fp: 1, fn: 2, tn: 4 };
  assert.ok(Math.abs(precision(cm) - 3 / 4) < 1e-12); // 0.75
  assert.ok(Math.abs(recall(cm) - 3 / 5) < 1e-12); // 0.6
  // f1 = 2*0.75*0.6/(0.75+0.6) = 0.9/1.35 = 0.6667
  assert.ok(Math.abs(f1(cm) - (2 * 0.75 * 0.6) / (0.75 + 0.6)) < 1e-12);
  assert.ok(Math.abs(accuracy(cm) - 7 / 10) < 1e-12);
});

test('confusionFromScores thresholds correctly', () => {
  const ex = [
    { s: 0.9, y: 1 },
    { s: 0.6, y: 1 },
    { s: 0.4, y: 0 },
    { s: 0.2, y: 0 },
    { s: 0.55, y: 0 }, // a false positive at thr 0.5
  ];
  const cm = confusionFromScores(ex, 0.5);
  assert.deepEqual(cm, { tp: 2, fp: 1, fn: 0, tn: 2 });
});

test('mse matches hand computation', () => {
  // errors: 1, -1, 0 -> squared 1,1,0 -> mean = 2/3
  const mseVal = mse([1, 2, 3], [2, 1, 3]);
  assert.ok(Math.abs(mseVal - 2 / 3) < 1e-12);
});

test('r2 is 1 for a perfect fit and 0 for predicting the mean', () => {
  assert.equal(r2([1, 2, 3], [1, 2, 3]), 1);
  // predicting the mean (2) for every point => SS_res == SS_tot => R2 = 0
  assert.ok(Math.abs(r2([1, 2, 3], [2, 2, 2]) - 0) < 1e-12);
});

test('r2 on a constructed partial fit', () => {
  // yTrue = [1,2,3] (mean 2, SS_tot = 1+0+1 = 2)
  // yPred = [1,2,2] (residuals 0,0,1 -> SS_res = 1) -> R2 = 1 - 1/2 = 0.5
  assert.ok(Math.abs(r2([1, 2, 3], [1, 2, 2]) - 0.5) < 1e-12);
});

test('rocAuc: perfectly separated scores give AUC 1', () => {
  const pos = [0.8, 0.9, 0.95];
  const neg = [0.1, 0.2, 0.3];
  const { auc } = rocAuc(pos, neg);
  assert.ok(Math.abs(auc - 1) < 1e-9, `auc ${auc}`);
});

test('rocAuc: identical populations give AUC ~ 0.5', () => {
  const same = [0.4, 0.5, 0.6];
  const { auc } = rocAuc(same, same);
  assert.ok(Math.abs(auc - 0.5) < 0.05, `auc ${auc}`);
});
