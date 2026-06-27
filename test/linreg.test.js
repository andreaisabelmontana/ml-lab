import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mseLoss,
  gradientDescentStep,
  fitLinearGD,
  polyfit,
  evalPoly,
  rmse,
  solve,
} from '../src/linreg.js';

test('recovers a known slope/intercept on noiseless data', () => {
  const W = 1.4, B = -0.3;
  const data = [];
  for (let i = 0; i < 50; i++) {
    const x = (i / 49) * 2 - 1; // x in [-1, 1]
    data.push({ x, y: W * x + B });
  }
  const { w, b } = fitLinearGD(data, { eta: 0.3, steps: 2000 });
  assert.ok(Math.abs(w - W) < 1e-3, `w=${w} should be ~${W}`);
  assert.ok(Math.abs(b - B) < 1e-3, `b=${b} should be ~${B}`);
});

test('gradient descent reduces loss at every step (convex MSE)', () => {
  const data = [];
  for (let i = 0; i < 30; i++) {
    const x = (i / 29) * 2 - 1;
    data.push({ x, y: 0.8 * x + 0.2 });
  }
  const { history } = fitLinearGD(data, { eta: 0.1, steps: 300 });
  for (let i = 1; i < history.length; i++) {
    assert.ok(
      history[i] <= history[i - 1] + 1e-12,
      `loss rose at step ${i}: ${history[i - 1]} -> ${history[i]}`
    );
  }
  // and it actually decreased overall
  assert.ok(history[history.length - 1] < history[0]);
});

test('one GD step moves parameters in the descent direction', () => {
  const data = [{ x: 1, y: 1 }, { x: 2, y: 2 }, { x: 3, y: 3 }];
  const before = mseLoss(data, 0, 0);
  const { w, b } = gradientDescentStep(data, 0, 0, 0.05);
  const after = mseLoss(data, w, b);
  assert.ok(after < before);
});

test('mseLoss matches hand computation', () => {
  const data = [{ x: 0, y: 1 }, { x: 1, y: 1 }];
  // w=0,b=0 -> errors are (-1),(-1); squared 1,1; mean = 1
  assert.equal(mseLoss(data, 0, 0), 1);
});

test('polyfit recovers an exact polynomial', () => {
  // y = 2 - 3x + x^2 sampled exactly
  const truth = (x) => 2 - 3 * x + x * x;
  const data = [-2, -1, 0, 1, 2, 3].map((x) => ({ x, y: truth(x) }));
  const co = polyfit(data, 2);
  assert.ok(Math.abs(co[0] - 2) < 1e-6);
  assert.ok(Math.abs(co[1] - -3) < 1e-6);
  assert.ok(Math.abs(co[2] - 1) < 1e-6);
  // and evaluating reproduces the data
  for (const p of data) assert.ok(Math.abs(evalPoly(co, p.x) - p.y) < 1e-6);
  // exact fit => zero RMSE
  assert.ok(rmse(data, co) < 1e-6);
});

test('solve solves a small linear system', () => {
  // 2x + y = 5 ; x + 3y = 10  -> x = 1, y = 3
  const x = solve([[2, 1], [1, 3]], [5, 10]);
  assert.ok(Math.abs(x[0] - 1) < 1e-9);
  assert.ok(Math.abs(x[1] - 3) < 1e-9);
});
