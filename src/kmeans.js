// k-means clustering via Lloyd's algorithm on 2-D points.
//
// `assignStep` labels each point with its nearest centroid; `updateStep` moves
// each centroid to the mean of its members and reports how far the centroids
// travelled. `inertia` is the within-cluster sum of squared distances (WCSS),
// which Lloyd's iteration is guaranteed not to increase. `fit` runs to
// convergence (or a step cap) from given initial centroids.

// Returns an assignment array: assign[i] = index of nearest centroid.
export function assignStep(pts, cents) {
  return pts.map((p) => {
    let bd = Infinity,
      bj = 0;
    cents.forEach((c, j) => {
      const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
      if (d < bd) {
        bd = d;
        bj = j;
      }
    });
    return bj;
  });
}

// Moves each centroid to the mean of its assigned points. Empty clusters stay
// put. Returns { cents, moved } where `moved` is the total centroid travel.
export function updateStep(pts, cents, assign) {
  let moved = 0;
  const next = cents.map((c, j) => {
    const mem = pts.filter((_, i) => assign[i] === j);
    if (!mem.length) return { x: c.x, y: c.y };
    const nx = mem.reduce((a, p) => a + p.x, 0) / mem.length;
    const ny = mem.reduce((a, p) => a + p.y, 0) / mem.length;
    moved += Math.hypot(nx - c.x, ny - c.y);
    return { x: nx, y: ny };
  });
  return { cents: next, moved };
}

// Within-cluster sum of squared distances for the current assignment.
export function inertia(pts, cents, assign) {
  let s = 0;
  pts.forEach((p, i) => {
    const c = cents[assign[i]];
    if (c) s += (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
  });
  return s;
}

// Run Lloyd's algorithm from `init` centroids until the centroids stop moving
// (below `tol`) or `maxIter` is reached. Returns the final centroids,
// assignment, iteration count and inertia.
export function fit(pts, init, { maxIter = 100, tol = 1e-9 } = {}) {
  let cents = init.map((c) => ({ x: c.x, y: c.y }));
  let assign = assignStep(pts, cents);
  let iter = 0;
  for (; iter < maxIter; iter++) {
    assign = assignStep(pts, cents);
    const res = updateStep(pts, cents, assign);
    cents = res.cents;
    if (res.moved < tol) {
      iter++;
      break;
    }
  }
  assign = assignStep(pts, cents);
  return { cents, assign, iter, inertia: inertia(pts, cents, assign) };
}
