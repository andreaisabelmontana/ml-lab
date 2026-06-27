// Feature scaling for 2-D point clouds [{x, y}].
//
//  - `columnStats` reports mean, (population) standard deviation and range per
//    axis.
//  - `standardize` applies the z-score transform z = (x - mu) / sigma, giving
//    each axis zero mean and unit variance.
//  - `minMax` maps each axis onto [0, 1].

export function columnStats(arr, key) {
  const n = arr.length || 1;
  const m = arr.reduce((a, p) => a + p[key], 0) / n;
  const sd = Math.sqrt(arr.reduce((a, p) => a + (p[key] - m) ** 2, 0) / n);
  const lo = Math.min(...arr.map((p) => p[key]));
  const hi = Math.max(...arr.map((p) => p[key]));
  return { m, sd, lo, hi };
}

// z-score each axis: zero mean, unit standard deviation.
export function standardize(raw) {
  const sx = columnStats(raw, 'x');
  const sy = columnStats(raw, 'y');
  return raw.map((p) => ({
    x: sx.sd ? (p.x - sx.m) / sx.sd : 0,
    y: sy.sd ? (p.y - sy.m) / sy.sd : 0,
  }));
}

// Min–max each axis onto [0, 1].
export function minMax(raw) {
  const sx = columnStats(raw, 'x');
  const sy = columnStats(raw, 'y');
  const dx = sx.hi - sx.lo || 1;
  const dy = sy.hi - sy.lo || 1;
  return raw.map((p) => ({
    x: (p.x - sx.lo) / dx,
    y: (p.y - sy.lo) / dy,
  }));
}
