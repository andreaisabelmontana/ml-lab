// Principal component analysis for a 2-D point cloud [{x, y}].
//
// `covariance2` builds the 2x2 covariance matrix; `eig2` is a closed-form
// symmetric 2x2 eigen-decomposition; `pca` ties them together and reports the
// principal axes, eigenvalues and the variance explained by the first
// component.

// Population covariance entries [cxx, cxy, cyy] of the cloud.
export function covariance2(raw) {
  const n = raw.length || 1;
  const mx = raw.reduce((a, p) => a + p.x, 0) / n;
  const my = raw.reduce((a, p) => a + p.y, 0) / n;
  let cxx = 0,
    cyy = 0,
    cxy = 0;
  raw.forEach((p) => {
    cxx += (p.x - mx) ** 2;
    cyy += (p.y - my) ** 2;
    cxy += (p.x - mx) * (p.y - my);
  });
  return { mx, my, cxx: cxx / n, cyy: cyy / n, cxy: cxy / n };
}

// Eigen-decomposition of the symmetric matrix [[a, b], [b, c]].
// Returns { l1 >= l2, v1 } where v1 is the (unit) eigenvector for l1.
export function eig2(a, b, c) {
  const tr = a + c;
  const det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  let vx, vy;
  if (Math.abs(b) > 1e-9) {
    vx = l1 - c;
    vy = b;
  } else {
    vx = 1;
    vy = 0;
  }
  const nrm = Math.hypot(vx, vy) || 1;
  return { l1, l2, v1: { x: vx / nrm, y: vy / nrm } };
}

// Full PCA of the cloud. Returns mean, eigenvalues, first principal axis and
// the fraction of variance kept by projecting onto PC1.
export function pca(raw) {
  const { mx, my, cxx, cxy, cyy } = covariance2(raw);
  const { l1, l2, v1 } = eig2(cxx, cxy, cyy);
  const v2 = { x: -v1.y, y: v1.x };
  const kept = l1 + l2 ? l1 / (l1 + l2) : 0;
  return { mx, my, l1, l2, v1, v2, kept };
}
