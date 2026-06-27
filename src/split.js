// Train/test split — partition the indices 0..n-1 into a training set and a
// held-out test set, reproducibly when given a seeded generator.
//
// The split is a true partition: every index lands in exactly one side, the
// two sides never overlap, and together they cover all n indices. The size of
// the training side is round(n * trainFraction), clamped so that both sides
// keep at least one element whenever n >= 2.

import { mulberry32 } from './rng.js';

// Fisher–Yates shuffle of [0, 1, ..., n-1] using a uniform generator.
export function shuffledIndices(n, rng) {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx;
}

// Returns { train, test } index arrays. `seed` makes the partition
// deterministic; `trainFraction` is in [0, 1].
export function trainTestSplit(n, { trainFraction = 0.6, seed = 1 } = {}) {
  if (!Number.isInteger(n) || n < 0) {
    throw new RangeError('n must be a non-negative integer');
  }
  if (n === 0) return { train: [], test: [] };

  const rng = mulberry32(seed);
  const order = shuffledIndices(n, rng);

  let trainSize = Math.round(n * trainFraction);
  if (n >= 2) {
    // keep at least one example on each side
    trainSize = Math.max(1, Math.min(n - 1, trainSize));
  } else {
    trainSize = Math.max(0, Math.min(n, trainSize));
  }

  return {
    train: order.slice(0, trainSize),
    test: order.slice(trainSize),
  };
}
