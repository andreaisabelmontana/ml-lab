import test from 'node:test';
import assert from 'node:assert/strict';
import { covariance2, eig2, pca } from '../src/pca.js';

test('eig2 on a diagonal matrix returns the diagonal as eigenvalues', () => {
  const { l1, l2, v1 } = eig2(3, 0, 1);
  assert.ok(Math.abs(l1 - 3) < 1e-12);
  assert.ok(Math.abs(l2 - 1) < 1e-12);
  // largest variance is along x
  assert.ok(Math.abs(Math.abs(v1.x) - 1) < 1e-12);
  assert.ok(Math.abs(v1.y) < 1e-12);
});

test('eigenvector is unit length and an actual eigenvector', () => {
  const a = 2, b = 1, c = 2;
  const { l1, v1 } = eig2(a, b, c);
  assert.ok(Math.abs(Math.hypot(v1.x, v1.y) - 1) < 1e-12);
  // A v should equal l1 v
  const Av = { x: a * v1.x + b * v1.y, y: b * v1.x + c * v1.y };
  assert.ok(Math.abs(Av.x - l1 * v1.x) < 1e-9);
  assert.ok(Math.abs(Av.y - l1 * v1.y) < 1e-9);
});

test('covariance2 matches hand computation', () => {
  const raw = [
    { x: 0, y: 0 },
    { x: 2, y: 0 },
    { x: 0, y: 2 },
    { x: 2, y: 2 },
  ];
  const { mx, my, cxx, cyy, cxy } = covariance2(raw);
  assert.equal(mx, 1);
  assert.equal(my, 1);
  assert.equal(cxx, 1); // population var of {0,2,0,2} about mean 1 = 1
  assert.equal(cyy, 1);
  assert.equal(cxy, 0); // uncorrelated
});

test('pca: first axis aligns with the dominant direction', () => {
  // points stretched along the x-axis
  const raw = [
    { x: -3, y: 0 },
    { x: -1, y: 0.1 },
    { x: 1, y: -0.1 },
    { x: 3, y: 0 },
  ];
  const { l1, l2, v1, kept } = pca(raw);
  assert.ok(l1 > l2);
  assert.ok(Math.abs(Math.abs(v1.x) - 1) < 0.05); // nearly horizontal
  assert.ok(kept > 0.9); // most variance on PC1
});
