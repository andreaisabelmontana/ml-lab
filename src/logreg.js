// Logistic regression — a 2-feature linear classifier trained by batch
// gradient descent on cross-entropy loss. Predicts p = sigmoid(w1 x + w2 y + b).

export function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// Predicted positive-class probability for a point {x, y}.
export function predictProba({ w1, w2, b }, p) {
  return sigmoid(w1 * p.x + w2 * p.y + b);
}

// One epoch of batch gradient descent. Data points are {x, y, t} with target
// t in {0, 1}. The gradient of mean cross-entropy w.r.t. the logit is just
// (p - t), so dL/dw1 = (1/m) Σ (p - t) x, etc. Returns updated params.
export function logisticStep(params, data, eta) {
  const { w1, w2, b } = params;
  const m = data.length || 1;
  let g1 = 0,
    g2 = 0,
    gb = 0;
  data.forEach((p) => {
    const e = sigmoid(w1 * p.x + w2 * p.y + b) - p.t;
    g1 += e * p.x;
    g2 += e * p.y;
    gb += e;
  });
  return {
    w1: w1 - (eta * g1) / m,
    w2: w2 - (eta * g2) / m,
    b: b - (eta * gb) / m,
  };
}

// Mean cross-entropy loss and 0/1-threshold accuracy over the data.
export function logisticMetrics(params, data) {
  if (!data.length) return { loss: 0, acc: 0 };
  let loss = 0,
    correct = 0;
  data.forEach((p) => {
    const yh = Math.min(1 - 1e-9, Math.max(1e-9, predictProba(params, p)));
    loss += -(p.t * Math.log(yh) + (1 - p.t) * Math.log(1 - yh));
    if ((yh >= 0.5 ? 1 : 0) === p.t) correct++;
  });
  return { loss: loss / data.length, acc: correct / data.length };
}

// Train from zero for `epochs` epochs. Returns final params + loss history.
export function fitLogistic(data, { eta = 0.3, epochs = 200 } = {}) {
  let params = { w1: 0, w2: 0, b: 0 };
  const history = [logisticMetrics(params, data).loss];
  for (let i = 0; i < epochs; i++) {
    params = logisticStep(params, data, eta);
    history.push(logisticMetrics(params, data).loss);
  }
  return { ...params, history };
}
