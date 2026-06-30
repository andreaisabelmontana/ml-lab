import test from 'node:test';
import assert from 'node:assert/strict';
import {
  initMLP,
  forward,
  predictProba,
  trainStep,
  mlpMetrics,
  fitMLP,
} from '../src/mlp.js';
import { fitLogistic, logisticMetrics } from '../src/logreg.js';
import { mulberry32, gauss } from '../src/rng.js';

// XOR — the canonical non-linearly-separable problem: class 1 lives in the
// bottom-right and top-left quadrants, class 0 in the other two. No straight
// line separates it, so a linear classifier is stuck near chance.
function xorData(seed = 1) {
  const rng = mulberry32(seed);
  const data = [];
  const blob = (cx, cy, t) => {
    for (let i = 0; i < 30; i++) {
      data.push({ x: cx + gauss(rng) * 0.07, y: cy + gauss(rng) * 0.07, t });
    }
  };
  blob(0.25, 0.25, 0);
  blob(0.75, 0.25, 1);
  blob(0.25, 0.75, 1);
  blob(0.75, 0.75, 0);
  return data;
}

test('forward produces a probability in (0, 1)', () => {
  const net = initMLP(6, 3);
  const { p } = forward(net, { x: 0.4, y: 0.6 });
  assert.ok(p > 0 && p < 1, `p=${p} must lie in (0,1)`);
  assert.equal(predictProba(net, { x: 0.4, y: 0.6 }), p);
});

test('trainStep returns a new network and leaves the original untouched', () => {
  const net = initMLP(4, 7);
  const b2Before = net.b2;
  const next = trainStep(net, xorData(2), 0.5);
  assert.notEqual(next, net);
  assert.equal(net.b2, b2Before, 'original net must not be mutated');
});

test('learns XOR to high training accuracy (a linear model cannot)', () => {
  const data = xorData(11);

  const { net } = fitMLP(data, { hidden: 8, eta: 0.6, epochs: 2000, seed: 5 });
  const { acc } = mlpMetrics(net, data);
  assert.ok(acc >= 0.9, `MLP train accuracy ${acc} should reach >= 0.9 on XOR`);

  // The linear baseline is provably near chance on XOR.
  const lin = fitLogistic(data, { eta: 0.5, epochs: 2000 });
  const linAcc = logisticMetrics(lin, data).acc;
  assert.ok(linAcc < 0.75, `logistic baseline ${linAcc} should stay near chance on XOR`);
  assert.ok(acc > linAcc + 0.15, 'MLP should clearly beat the linear baseline');
});

test('training drives cross-entropy loss well below its starting value', () => {
  const { history } = fitMLP(xorData(3), { hidden: 8, eta: 0.6, epochs: 1500, seed: 1 });
  assert.ok(history[history.length - 1] < history[0] * 0.5, 'loss should at least halve');
});
