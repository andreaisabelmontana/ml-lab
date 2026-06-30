import test from 'node:test';
import assert from 'node:assert/strict';
import { kFoldPartition, crossValidate, cvCurve } from '../src/kfold.js';
import { mulberry32, gauss } from '../src/rng.js';

test('partition covers every index exactly once across k balanced folds', () => {
  const k = 5, n = 23;
  const folds = kFoldPartition(n, k, 7);
  assert.equal(folds.length, k);
  const all = folds.flat().sort((a, b) => a - b);
  assert.deepEqual(all, Array.from({ length: n }, (_, i) => i));
  const sizes = folds.map(f => f.length);
  assert.ok(Math.max(...sizes) - Math.min(...sizes) <= 1, `unbalanced: ${sizes}`);
});

test('a line is cross-validated with near-zero error by a degree-1 fit', () => {
  const pts = Array.from({ length: 20 }, (_, i) => ({ x: i / 19, y: 2 * (i / 19) - 0.5 }));
  const { mean, results } = crossValidate(pts, { degree: 1, k: 5, seed: 3 });
  assert.equal(results.length, 5);
  assert.ok(mean < 1e-6, `clean line should give ~0 CV error, got ${mean}`);
});

test('cross-validation exposes overfitting: a high-degree fit generalizes worse', () => {
  // noisy data from a gentle truth — a wild high-degree polynomial should have
  // a worse cross-validated error than a modest one.
  const rng = mulberry32(11);
  const truth = x => Math.sin(2 * Math.PI * x) * 0.5;
  const pts = Array.from({ length: 30 }, () => { const x = rng(); return { x, y: truth(x) + gauss(rng) * 0.15 }; });
  const lowCv = crossValidate(pts, { degree: 3, k: 5, seed: 2 }).mean;
  const highCv = crossValidate(pts, { degree: 14, k: 5, seed: 2 }).mean;
  assert.ok(highCv > lowCv, `degree-14 CV (${highCv}) should exceed degree-3 CV (${lowCv})`);
});

test('cvCurve returns one cross-validated error per requested degree', () => {
  const pts = Array.from({ length: 16 }, (_, i) => ({ x: i, y: i * i }));
  const curve = cvCurve(pts, [1, 2, 3], { k: 4, seed: 1 });
  assert.deepEqual(curve.map(c => c.degree), [1, 2, 3]);
  assert.ok(curve.every(c => Number.isFinite(c.cvRmse)));
});
