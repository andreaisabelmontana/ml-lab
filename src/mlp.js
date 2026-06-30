// Multi-layer perceptron — the smallest network that breaks the linear
// barrier: two inputs (x, y), one hidden layer of `tanh` units, and a single
// `sigmoid` output trained by full-batch gradient descent on cross-entropy.
// Pure and DOM-free, like the other cores under src/. Backprop here is the
// chain rule by hand: dL/dz2 = (ŷ − t) for sigmoid + cross-entropy, then the
// tanh derivative (1 − a²) carries the signal back into the hidden layer.

import { mulberry32, gauss } from './rng.js';

// A fresh network with `hidden` tanh units. Weights are small Gaussian draws
// (scaled by 1/√fan-in) so training starts in the near-linear regime and the
// symmetry between hidden units is broken. Deterministic given `seed`.
export function initMLP(hidden = 6, seed = 1) {
  const rng = mulberry32(seed);
  const s1 = 1 / Math.sqrt(2);
  const s2 = 1 / Math.sqrt(hidden);
  return {
    hidden,
    // W1[j] = [w_x, w_y] for hidden unit j; b1[j] its bias.
    W1: Array.from({ length: hidden }, () => [gauss(rng) * s1, gauss(rng) * s1]),
    b1: Array.from({ length: hidden }, () => 0),
    // W2[j] weights the j-th hidden activation into the output; b2 its bias.
    W2: Array.from({ length: hidden }, () => gauss(rng) * s2),
    b2: 0,
  };
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// Forward pass for one point {x, y}. Returns the output probability `p` plus
// the hidden activations `a1` (kept so backprop can reuse them).
export function forward(net, point) {
  const { W1, b1, W2, b2, hidden } = net;
  const a1 = new Array(hidden);
  let z2 = b2;
  for (let j = 0; j < hidden; j++) {
    const z1 = W1[j][0] * point.x + W1[j][1] * point.y + b1[j];
    const a = Math.tanh(z1);
    a1[j] = a;
    z2 += W2[j] * a;
  }
  return { p: sigmoid(z2), a1 };
}

// Probability the network assigns to the positive class.
export function predictProba(net, point) {
  return forward(net, point).p;
}

// One epoch of full-batch gradient descent. Data points are {x, y, t} with
// target t in {0, 1}. Returns a NEW network (the input is left untouched).
export function trainStep(net, data, eta) {
  const { W1, b1, W2, b2, hidden } = net;
  const m = data.length || 1;
  const gW1 = Array.from({ length: hidden }, () => [0, 0]);
  const gb1 = new Array(hidden).fill(0);
  const gW2 = new Array(hidden).fill(0);
  let gb2 = 0;

  for (const pt of data) {
    const { p, a1 } = forward(net, pt);
    const dz2 = p - pt.t;            // ∂L/∂z2 for sigmoid + cross-entropy
    gb2 += dz2;
    for (let j = 0; j < hidden; j++) {
      gW2[j] += dz2 * a1[j];
      const dz1 = dz2 * W2[j] * (1 - a1[j] * a1[j]); // through tanh'
      gW1[j][0] += dz1 * pt.x;
      gW1[j][1] += dz1 * pt.y;
      gb1[j] += dz1;
    }
  }

  const next = {
    hidden,
    W1: W1.map((row, j) => [row[0] - (eta * gW1[j][0]) / m, row[1] - (eta * gW1[j][1]) / m]),
    b1: b1.map((bj, j) => bj - (eta * gb1[j]) / m),
    W2: W2.map((wj, j) => wj - (eta * gW2[j]) / m),
    b2: b2 - (eta * gb2) / m,
  };
  return next;
}

// Mean cross-entropy loss and 0/1-threshold accuracy over the data.
export function mlpMetrics(net, data) {
  if (!data.length) return { loss: 0, acc: 0 };
  let loss = 0;
  let correct = 0;
  for (const pt of data) {
    const yh = Math.min(1 - 1e-9, Math.max(1e-9, predictProba(net, pt)));
    loss += -(pt.t * Math.log(yh) + (1 - pt.t) * Math.log(1 - yh));
    if ((yh >= 0.5 ? 1 : 0) === pt.t) correct++;
  }
  return { loss: loss / data.length, acc: correct / data.length };
}

// Train from a fresh network for `epochs` epochs. Returns the final network
// plus the loss history — used by the tests and handy for experimentation.
export function fitMLP(data, { hidden = 6, eta = 0.5, epochs = 400, seed = 1 } = {}) {
  let net = initMLP(hidden, seed);
  const history = [mlpMetrics(net, data).loss];
  for (let i = 0; i < epochs; i++) {
    net = trainStep(net, data, eta);
    history.push(mlpMetrics(net, data).loss);
  }
  return { net, history };
}
