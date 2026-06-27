import test from 'node:test';
import assert from 'node:assert/strict';
import { classify, looAccuracy } from '../src/knn.js';
import { mulberry32, gauss } from '../src/rng.js';

function clusters(seed = 1) {
  const rng = mulberry32(seed);
  const train = [];
  for (let i = 0; i < 30; i++) {
    train.push({ x: 0.25 + gauss(rng) * 0.05, y: 0.25 + gauss(rng) * 0.05, c: 0 });
  }
  for (let i = 0; i < 30; i++) {
    train.push({ x: 0.75 + gauss(rng) * 0.05, y: 0.75 + gauss(rng) * 0.05, c: 1 });
  }
  return train;
}

test('classifies separable clusters correctly', () => {
  const train = clusters(5);
  // a point deep in cluster 0 territory
  assert.equal(classify(train, 0.25, 0.25, 5).cls, 0);
  // a point deep in cluster 1 territory
  assert.equal(classify(train, 0.75, 0.75, 5).cls, 1);
});

test('vote counts sum to k (when k <= n)', () => {
  const train = clusters(2);
  const { v0, v1 } = classify(train, 0.5, 0.5, 7);
  assert.equal(v0 + v1, 7);
});

test('leave-one-out accuracy is high on well-separated clusters', () => {
  const train = clusters(9);
  const acc = looAccuracy(train, 5);
  assert.ok(acc >= 0.95, `LOO accuracy ${acc} should be >= 0.95`);
});

test('k=1 nearest neighbour returns the exact closest point label', () => {
  const train = [
    { x: 0, y: 0, c: 0 },
    { x: 1, y: 1, c: 1 },
  ];
  assert.equal(classify(train, 0.1, 0.1, 1).cls, 0);
  assert.equal(classify(train, 0.9, 0.9, 1).cls, 1);
});
