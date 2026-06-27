import test from 'node:test';
import assert from 'node:assert/strict';
import { columnStats, standardize, minMax } from '../src/scaling.js';
import { mulberry32, gauss } from '../src/rng.js';

function cloud(seed = 1) {
  const rng = mulberry32(seed);
  const raw = [];
  for (let i = 0; i < 100; i++) {
    raw.push({ x: 50 + gauss(rng) * 18, y: 5 + gauss(rng) * 1.2 });
  }
  return raw;
}

test('standardize gives each axis zero mean and unit variance', () => {
  const z = standardize(cloud(4));
  const sx = columnStats(z, 'x');
  const sy = columnStats(z, 'y');
  assert.ok(Math.abs(sx.m) < 1e-9, `mean x ${sx.m}`);
  assert.ok(Math.abs(sy.m) < 1e-9, `mean y ${sy.m}`);
  assert.ok(Math.abs(sx.sd - 1) < 1e-9, `std x ${sx.sd}`);
  assert.ok(Math.abs(sy.sd - 1) < 1e-9, `std y ${sy.sd}`);
});

test('minMax maps each axis onto [0,1]', () => {
  const m = minMax(cloud(6));
  const sx = columnStats(m, 'x');
  const sy = columnStats(m, 'y');
  assert.ok(Math.abs(sx.lo - 0) < 1e-12);
  assert.ok(Math.abs(sx.hi - 1) < 1e-12);
  assert.ok(Math.abs(sy.lo - 0) < 1e-12);
  assert.ok(Math.abs(sy.hi - 1) < 1e-12);
  for (const p of m) {
    assert.ok(p.x >= 0 && p.x <= 1);
    assert.ok(p.y >= 0 && p.y <= 1);
  }
});

test('columnStats matches hand computation', () => {
  const arr = [{ x: 0 }, { x: 2 }, { x: 4 }];
  const s = columnStats(arr, 'x');
  assert.equal(s.m, 2);
  assert.equal(s.lo, 0);
  assert.equal(s.hi, 4);
  // population sd of {0,2,4} = sqrt((4+0+4)/3) = sqrt(8/3)
  assert.ok(Math.abs(s.sd - Math.sqrt(8 / 3)) < 1e-12);
});
