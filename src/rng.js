// Seedable pseudo-random utilities used across the demos.
// Pure, no DOM. `mulberry32` is a small, fast 32-bit generator; `gauss`
// draws a standard-normal sample from it via the Box–Muller transform.

// Returns a deterministic generator: same seed -> same sequence of
// uniform values in [0, 1).
export function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// One standard-normal sample (mean 0, variance 1) from a uniform generator.
export function gauss(rng) {
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
