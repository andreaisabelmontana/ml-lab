// Decision-tree primitives — Gini impurity and the greedy best axis-aligned
// split. Points are {x, y, c} with binary class c in {0, 1}. `bestSplit` scans
// every midpoint threshold on each axis and returns the split that most
// reduces weighted child impurity, or null if no positive-gain split exists.

// Gini impurity of a binary-labelled set: 1 - p1^2 - p0^2.
export function gini(pts) {
  if (!pts.length) return 0;
  const p1 = pts.filter((p) => p.c === 1).length / pts.length;
  return 1 - p1 * p1 - (1 - p1) * (1 - p1);
}

// Greedy best split. Returns { ax, thr, L, R, gain } or null.
export function bestSplit(pts) {
  let best = null,
    bestGain = 1e-9;
  const g0 = gini(pts);
  ['x', 'y'].forEach((ax) => {
    const vals = [...new Set(pts.map((p) => p[ax]))].sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i] + vals[i + 1]) / 2;
      const L = pts.filter((p) => p[ax] <= thr);
      const R = pts.filter((p) => p[ax] > thr);
      if (!L.length || !R.length) continue;
      const g = (L.length * gini(L) + R.length * gini(R)) / pts.length;
      const gain = g0 - g;
      if (gain > bestGain) {
        bestGain = gain;
        best = { ax, thr, L, R, gain };
      }
    }
  });
  return best;
}

// Build the leaf boxes of a depth-limited tree over the unit square. Each leaf
// is { x0, x1, y0, y1, cls, pts } where cls is the majority class. Returned via
// the `out` array so callers can collect every leaf.
export function buildTree(pts, box, depth, maxDepth, out) {
  const maj = pts.filter((p) => p.c === 1).length >= pts.length / 2 ? 1 : 0;
  if (depth >= maxDepth || gini(pts) < 1e-6 || pts.length < 3) {
    out.push({ ...box, cls: maj, pts });
    return;
  }
  const s = bestSplit(pts);
  if (!s) {
    out.push({ ...box, cls: maj, pts });
    return;
  }
  if (s.ax === 'x') {
    buildTree(s.L, { ...box, x1: s.thr }, depth + 1, maxDepth, out);
    buildTree(s.R, { ...box, x0: s.thr }, depth + 1, maxDepth, out);
  } else {
    buildTree(s.L, { ...box, y1: s.thr }, depth + 1, maxDepth, out);
    buildTree(s.R, { ...box, y0: s.thr }, depth + 1, maxDepth, out);
  }
}
