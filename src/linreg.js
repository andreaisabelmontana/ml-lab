// Linear regression, two ways.
//
//  - `gradientDescentStep` / `fitLinearGD`: 1-D linear regression y = w x + b
//    trained by batch gradient descent on mean-squared-error loss. This is the
//    engine behind the "gradient descent" demo.
//  - `polyfit` / `evalPoly`: least-squares polynomial fit via the normal
//    equations on a Vandermonde basis, solved by Gaussian elimination with
//    partial pivoting. This is the engine behind the "train/test split &
//    overfitting" demo.

// Mean-squared-error loss of the line (w, b) over points [{x, y}].
export function mseLoss(data, w, b) {
  if (!data.length) return 0;
  return data.reduce((a, p) => a + (w * p.x + b - p.y) ** 2, 0) / data.length;
}

// One batch gradient-descent step. Returns the updated { w, b }.
// Gradient of J = (1/m) Σ (w x + b - y)^2 is
//   dJ/dw = (2/m) Σ (w x + b - y) x,   dJ/db = (2/m) Σ (w x + b - y).
export function gradientDescentStep(data, w, b, eta) {
  const m = data.length || 1;
  let gw = 0,
    gb = 0;
  data.forEach((p) => {
    const e = w * p.x + b - p.y;
    gw += 2 * e * p.x;
    gb += 2 * e;
  });
  gw /= m;
  gb /= m;
  return { w: w - eta * gw, b: b - eta * gb };
}

// Run `steps` of gradient descent from (w0, b0). Returns the final parameters
// plus the loss history (length steps + 1, including the initial loss).
export function fitLinearGD(
  data,
  { eta = 0.1, steps = 200, w0 = 0, b0 = 0 } = {}
) {
  let w = w0,
    b = b0;
  const history = [mseLoss(data, w, b)];
  for (let i = 0; i < steps; i++) {
    ({ w, b } = gradientDescentStep(data, w, b, eta));
    history.push(mseLoss(data, w, b));
  }
  return { w, b, history };
}

// Solve the linear system A x = b in place via Gaussian elimination with
// partial pivoting. A is an m x m array of rows; b is length m.
export function solve(A, b) {
  const m = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < m; c++) {
    let piv = c;
    for (let r = c + 1; r < m; r++) {
      if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    }
    const t = M[c];
    M[c] = M[piv];
    M[piv] = t;
    if (Math.abs(M[c][c]) < 1e-12) M[c][c] = 1e-12;
    for (let r = 0; r < m; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= m; k++) M[r][k] -= f * M[c][k];
    }
  }
  const out = [];
  for (let i = 0; i < m; i++) out.push(M[i][m] / M[i][i]);
  return out;
}

// Least-squares polynomial fit of degree `deg`. Builds the normal-equation
// system from the Vandermonde basis and solves it. Returns coefficients
// [c0, c1, ..., cdeg] for c0 + c1 x + ... + cdeg x^deg.
export function polyfit(data, deg) {
  const m = deg + 1;
  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  const bv = new Array(m).fill(0);
  data.forEach(({ x, y }) => {
    const pw = [1];
    for (let k = 1; k < 2 * deg + 1; k++) pw.push(pw[k - 1] * x);
    for (let i = 0; i < m; i++) {
      bv[i] += pw[i] * y;
      for (let j = 0; j < m; j++) A[i][j] += pw[i + j];
    }
  });
  return solve(A, bv);
}

// Evaluate a polynomial (coefficient array, ascending powers) at x.
export function evalPoly(co, x) {
  let s = 0,
    p = 1;
  for (let i = 0; i < co.length; i++) {
    s += co[i] * p;
    p *= x;
  }
  return s;
}

// Root-mean-squared error of a polynomial fit over [{x, y}].
export function rmse(data, co) {
  if (!data.length) return 0;
  return Math.sqrt(
    data.reduce((a, p) => a + (evalPoly(co, p.x) - p.y) ** 2, 0) / data.length
  );
}
