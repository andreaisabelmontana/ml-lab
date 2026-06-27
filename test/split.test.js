import test from 'node:test';
import assert from 'node:assert/strict';
import { trainTestSplit, shuffledIndices } from '../src/split.js';
import { mulberry32 } from '../src/rng.js';

test('split is a partition: no overlap, full coverage, right sizes', () => {
  const n = 100;
  const { train, test } = trainTestSplit(n, { trainFraction: 0.7, seed: 1 });
  assert.equal(train.length, 70);
  assert.equal(test.length, 30);

  const trainSet = new Set(train);
  const testSet = new Set(test);
  // no overlap
  for (const i of train) assert.ok(!testSet.has(i));
  // no duplicates within sides
  assert.equal(trainSet.size, train.length);
  assert.equal(testSet.size, test.length);
  // together they cover exactly 0..n-1
  const all = new Set([...train, ...test]);
  assert.equal(all.size, n);
  for (let i = 0; i < n; i++) assert.ok(all.has(i));
});

test('split is reproducible for the same seed and varies across seeds', () => {
  const a = trainTestSplit(50, { trainFraction: 0.6, seed: 7 });
  const b = trainTestSplit(50, { trainFraction: 0.6, seed: 7 });
  const c = trainTestSplit(50, { trainFraction: 0.6, seed: 8 });
  assert.deepEqual(a.train, b.train);
  assert.deepEqual(a.test, b.test);
  assert.notDeepEqual(a.train, c.train);
});

test('train size = round(n * fraction)', () => {
  assert.equal(trainTestSplit(10, { trainFraction: 0.6, seed: 1 }).train.length, 6);
  assert.equal(trainTestSplit(10, { trainFraction: 0.25, seed: 1 }).train.length, 3);
  assert.equal(trainTestSplit(7, { trainFraction: 0.5, seed: 1 }).train.length, 4); // round(3.5)=4
});

test('both sides keep at least one element when n >= 2', () => {
  const lo = trainTestSplit(5, { trainFraction: 0, seed: 3 });
  assert.equal(lo.train.length, 1);
  assert.equal(lo.test.length, 4);
  const hi = trainTestSplit(5, { trainFraction: 1, seed: 3 });
  assert.equal(hi.train.length, 4);
  assert.equal(hi.test.length, 1);
});

test('edge cases: n = 0 and n = 1', () => {
  assert.deepEqual(trainTestSplit(0, { seed: 1 }), { train: [], test: [] });
  const one = trainTestSplit(1, { trainFraction: 0.6, seed: 1 });
  assert.equal(one.train.length + one.test.length, 1);
});

test('shuffledIndices is a permutation of 0..n-1', () => {
  const idx = shuffledIndices(20, mulberry32(99));
  assert.equal(idx.length, 20);
  assert.deepEqual([...idx].sort((a, b) => a - b), Array.from({ length: 20 }, (_, i) => i));
});

test('rejects invalid n', () => {
  assert.throws(() => trainTestSplit(-1, {}), RangeError);
  assert.throws(() => trainTestSplit(2.5, {}), RangeError);
});
