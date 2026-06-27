import test from 'node:test';
import assert from 'node:assert/strict';
import {
  sigmoid,
  logisticStep,
  logisticMetrics,
  fitLogistic,
} from '../src/logreg.js';
import { mulberry32, gauss } from '../src/rng.js';

test('sigmoid basics', () => {
  assert.equal(sigmoid(0), 0.5);
  assert.ok(sigmoid(10) > 0.99);
  assert.ok(sigmoid(-10) < 0.01);
});

function separableData(seed = 1) {
  const rng = mulberry32(seed);
  const data = [];
  for (let i = 0; i < 40; i++) {
    data.push({ x: 0.25 + gauss(rng) * 0.06, y: 0.25 + gauss(rng) * 0.06, t: 0 });
  }
  for (let i = 0; i < 40; i++) {
    data.push({ x: 0.75 + gauss(rng) * 0.06, y: 0.75 + gauss(rng) * 0.06, t: 1 });
  }
  return data;
}

test('separates a linearly separable set with high train accuracy', () => {
  const data = separableData(11);
  const params = fitLogistic(data, { eta: 0.5, epochs: 1500 });
  const { acc } = logisticMetrics(params, data);
  assert.ok(acc >= 0.95, `train accuracy ${acc} should be >= 0.95`);
});

test('cross-entropy loss decreases monotonically during training', () => {
  const data = separableData(3);
  const params = fitLogistic(data, { eta: 0.3, epochs: 400 });
  const h = params.history;
  for (let i = 1; i < h.length; i++) {
    assert.ok(h[i] <= h[i - 1] + 1e-9, `loss rose at epoch ${i}`);
  }
  assert.ok(h[h.length - 1] < h[0]);
});

test('one step reduces loss on a tiny set', () => {
  const data = [
    { x: 0, y: 0, t: 0 },
    { x: 1, y: 1, t: 1 },
  ];
  const before = logisticMetrics({ w1: 0, w2: 0, b: 0 }, data).loss;
  const p = logisticStep({ w1: 0, w2: 0, b: 0 }, data, 0.5);
  const after = logisticMetrics(p, data).loss;
  assert.ok(after < before);
});
