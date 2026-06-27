import test from 'node:test';
import assert from 'node:assert/strict';
import { gini, bestSplit, buildTree } from '../src/tree.js';

test('gini is 0 for a pure set and 0.5 for a 50/50 split', () => {
  assert.equal(gini([{ c: 1 }, { c: 1 }]), 0);
  assert.equal(gini([{ c: 0 }, { c: 0 }]), 0);
  assert.equal(gini([{ c: 0 }, { c: 1 }]), 0.5);
});

test('bestSplit finds the clean axis-aligned boundary', () => {
  // class 0 on the left (x < 0.5), class 1 on the right
  const pts = [
    { x: 0.1, y: 0.5, c: 0 },
    { x: 0.2, y: 0.5, c: 0 },
    { x: 0.8, y: 0.5, c: 1 },
    { x: 0.9, y: 0.5, c: 1 },
  ];
  const s = bestSplit(pts);
  assert.equal(s.ax, 'x');
  assert.ok(s.thr > 0.2 && s.thr < 0.8);
  // the split is perfect: both children pure
  assert.equal(gini(s.L), 0);
  assert.equal(gini(s.R), 0);
  assert.ok(s.gain > 0);
});

test('bestSplit returns null when no split helps', () => {
  const pts = [
    { x: 0.1, y: 0.1, c: 0 },
    { x: 0.1, y: 0.1, c: 1 },
  ];
  assert.equal(bestSplit(pts), null);
});

test('buildTree perfectly separates a linearly separable set', () => {
  const pts = [
    { x: 0.1, y: 0.5, c: 0 },
    { x: 0.2, y: 0.5, c: 0 },
    { x: 0.3, y: 0.5, c: 0 },
    { x: 0.7, y: 0.5, c: 1 },
    { x: 0.8, y: 0.5, c: 1 },
    { x: 0.9, y: 0.5, c: 1 },
  ];
  const leaves = [];
  buildTree(pts, { x0: 0, x1: 1, y0: 0, y1: 1 }, 0, 4, leaves);
  let correct = 0;
  leaves.forEach((L) => {
    correct += L.pts.filter((p) => p.c === L.cls).length;
  });
  assert.equal(correct, pts.length); // 100% train accuracy
});
