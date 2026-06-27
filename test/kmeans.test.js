import test from 'node:test';
import assert from 'node:assert/strict';
import { assignStep, updateStep, inertia, fit } from '../src/kmeans.js';
import { mulberry32, gauss } from '../src/rng.js';

// Three tight, well-separated blobs.
function blobs(seed = 1) {
  const rng = mulberry32(seed);
  const centers = [
    { x: 0.2, y: 0.2 },
    { x: 0.8, y: 0.2 },
    { x: 0.5, y: 0.8 },
  ];
  const pts = [];
  for (const c of centers) {
    for (let i = 0; i < 30; i++) {
      pts.push({ x: c.x + gauss(rng) * 0.03, y: c.y + gauss(rng) * 0.03 });
    }
  }
  return { pts, centers };
}

test('converges to the known blob centers', () => {
  const { pts, centers } = blobs(7);
  // initialise near (but not on) the true centers
  const init = centers.map((c) => ({ x: c.x + 0.04, y: c.y - 0.04 }));
  const { cents, iter } = fit(pts, init, { maxIter: 100 });
  assert.ok(iter < 100, 'should converge before the cap');
  // each true center should be matched by some recovered centroid
  for (const tc of centers) {
    const near = cents.some(
      (c) => Math.hypot(c.x - tc.x, c.y - tc.y) < 0.05
    );
    assert.ok(near, `no centroid near true center ${JSON.stringify(tc)}`);
  }
});

test('inertia is non-increasing across Lloyd iterations', () => {
  const { pts } = blobs(3);
  let cents = [
    { x: 0.1, y: 0.1 },
    { x: 0.9, y: 0.5 },
    { x: 0.4, y: 0.9 },
  ];
  let prev = Infinity;
  for (let step = 0; step < 20; step++) {
    const assign = assignStep(pts, cents);
    const cur = inertia(pts, cents, assign);
    assert.ok(cur <= prev + 1e-9, `inertia rose at step ${step}: ${prev} -> ${cur}`);
    prev = cur;
    cents = updateStep(pts, cents, assign).cents;
  }
});

test('assignStep picks the nearest centroid', () => {
  const pts = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  const cents = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  assert.deepEqual(assignStep(pts, cents), [0, 1]);
});

test('updateStep moves a centroid to the mean of its members', () => {
  const pts = [{ x: 0, y: 0 }, { x: 2, y: 4 }];
  const cents = [{ x: 5, y: 5 }];
  const { cents: next } = updateStep(pts, cents, [0, 0]);
  assert.equal(next[0].x, 1);
  assert.equal(next[0].y, 2);
});
