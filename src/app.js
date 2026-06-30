// ============================================================
// ml-lab — 13 visual demos covering the foundational machine-learning
// pipeline and core algorithms taught in AI: Machine Learning Foundations
// (IE BCSAI): train/test split & overfitting, gradient descent,
// bias–variance, feature scaling, k-NN, logistic regression, decision
// trees, k-means, PCA, confusion matrix, ROC/AUC, a neural network
// (multi-layer perceptron), and k-fold cross-validation.
//
// Every demo follows the same pattern as the rest of the *-lab series:
//   1. read control state through helpers that always return finite values
//   2. compute into a local buffer
//   3. render in a single idempotent `draw()` that resets the transform
//      and clears before drawing, so resizes and rapid input never compound
//
// Animated demos use requestAnimationFrame with play/pause/reset and tear
// down the frame on reset. KaTeX renders the math in the captions.
// ============================================================
//
// The framework-free ML cores live in the sibling modules under src/ and are
// unit-tested with `node --test` (see test/). This file imports them and is
// responsible only for reading controls, driving the demos and rendering.

import { mulberry32, gauss } from './rng.js';
import { trainTestSplit } from './split.js';
import {
  mseLoss,
  gradientDescentStep,
  polyfit,
  evalPoly,
  rmse,
} from './linreg.js';
import { sigmoid, logisticStep, logisticMetrics } from './logreg.js';
import { classify as knnClassify, looAccuracy } from './knn.js';
import {
  assignStep,
  updateStep,
  inertia as kmInertia,
} from './kmeans.js';
import { gini, buildTree } from './tree.js';
import { columnStats, standardize, minMax } from './scaling.js';
import {
  confusionFromScores,
  accuracy as cmAccuracy,
  precision as cmPrecision,
  recall as cmRecall,
  f1 as cmF1,
  rocAuc,
} from './metrics.js';
import { pca as pcaFit } from './pca.js';
import { initMLP, forward, trainStep, mlpMetrics } from './mlp.js';
import { crossValidate } from './kfold.js';

// ---------- helpers ------------------------------------------------------
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function n(id, fallback) {
  const el = document.getElementById(id);
  const v = el ? +el.value : NaN;
  return Number.isFinite(v) ? v : fallback;
}
const $ = id => document.getElementById(id);
const setText = (id, t) => { const el = $(id); if (el) el.textContent = t; };

// ---------- palette ------------------------------------------------------
const ACCENT = '#4338CA';
const ACCENT_S = 'rgba(67,56,202,0.16)';
const RULE  = '#E5E5EA';
const RULE_H = '#CDCDD4';
const INK   = '#15151A';
const INK_S = '#4B4B55';
const MUTED = '#8A8A92';
const GOOD  = '#16A34A';
const WARN  = '#F59E0B';
const BAD   = '#DC2626';

function fitCanvas(cv) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = cv.getBoundingClientRect();
  const cssW = Math.max(80, rect.width);
  const cssH = Math.max(80, parseInt(cv.getAttribute('height'), 10) || 280);
  cv.width  = Math.floor(cssW * dpr);
  cv.height = Math.floor(cssH * dpr);
  cv.style.height = cssH + 'px';
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = '12px Inter, sans-serif';
  ctx.textBaseline = 'alphabetic';
  return { ctx, w: cssW, h: cssH };
}
// pointer position in CSS pixels relative to canvas
function ptr(cv, ev) {
  const r = cv.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}
// (mulberry32 + gauss are imported from ./rng.js — the same seedable RNG the
//  "new sample" buttons use to give reproducible-but-varied data.)

// ============================================================
// 1. TRAIN/TEST SPLIT & OVERFITTING — polynomial least squares
// ============================================================
(function trainTest() {
  const cv = $('cv-split'); if (!cv) return;
  let seed = 7, pts = [];
  // ground truth on x in [0,1]
  const truth = x => Math.sin(2 * Math.PI * x) * 0.6 + 0.1;

  function regen() {
    const rng = mulberry32(seed);
    const sigma = n('sp-n', 25) / 100;
    pts = [];
    for (let i = 0; i < 40; i++) {
      const x = rng();
      const y = truth(x) + gauss(rng) * sigma;
      pts.push({ x, y, test: false });
    }
    // partition into train/test reproducibly (seeded by the demo seed)
    const frac = n('sp-f', 60) / 100;
    const { test } = trainTestSplit(pts.length, { trainFraction: frac, seed });
    const testSet = new Set(test);
    pts.forEach((p, i) => { p.test = testSet.has(i); });
  }

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const deg = n('sp-d', 3);
    setText('sp-dv', deg);
    setText('sp-nv', (n('sp-n', 25) / 100).toFixed(2));
    setText('sp-fv', (n('sp-f', 60) / 100).toFixed(2));
    if (!pts.length) regen();

    const train = pts.filter(p => !p.test), test = pts.filter(p => p.test);
    const co = polyfit(train, deg);
    const trRMSE = rmse(train, co), teRMSE = rmse(test, co);

    // plot frame
    const pad = 34, x0 = pad, x1 = w - 12, y0 = 12, y1 = h - pad;
    const X = x => x0 + x * (x1 - x0);
    const Y = y => y1 - (y + 1.2) / 2.4 * (y1 - y0);
    ctx.strokeStyle = RULE; ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    // fitted curve
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = i / 200, y = clamp(evalPoly(co, x), -1.2, 1.2);
      const px = X(x), py = Y(y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // points
    train.forEach(p => { ctx.fillStyle = ACCENT; ctx.beginPath(); ctx.arc(X(p.x), Y(clamp(p.y, -1.2, 1.2)), 3.2, 0, 7); ctx.fill(); });
    test.forEach(p => {
      ctx.strokeStyle = WARN; ctx.lineWidth = 1.6; ctx.beginPath();
      ctx.arc(X(p.x), Y(clamp(p.y, -1.2, 1.2)), 4, 0, 7); ctx.stroke();
    });
    ctx.fillStyle = MUTED; ctx.font = '11px Inter';
    ctx.fillText('● train   ○ test', x0 + 6, y0 + 14);

    setText('sp-tr', trRMSE.toFixed(3));
    setText('sp-te', teRMSE.toFixed(3));
    const ratio = teRMSE / Math.max(1e-6, trRMSE);
    let verdict, col;
    if (ratio > 2.2 && deg >= 5) { verdict = 'overfitting'; col = BAD; }
    else if (trRMSE > 0.35) { verdict = 'underfitting'; col = WARN; }
    else { verdict = 'good fit'; col = GOOD; }
    setText('sp-v', verdict); $('sp-v').style.color = col;
  }
  $('sp-d').addEventListener('input', draw);
  $('sp-f').addEventListener('input', () => { regen(); draw(); });
  $('sp-n').addEventListener('input', () => { regen(); draw(); });
  $('sp-new').addEventListener('click', () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; regen(); draw(); });
  window.addEventListener('resize', draw);
  regen(); draw();
})();

// ============================================================
// 2. GRADIENT DESCENT — linear regression on noisy points
// ============================================================
(function gradientDescent() {
  const cv = $('cv-gd'); if (!cv) return;
  const data = [];
  const TRUE_W = 1.4, TRUE_B = -0.3;
  (function build() {
    const rng = mulberry32(11);
    for (let i = 0; i < 30; i++) {
      const x = rng() * 2 - 1;
      data.push({ x, y: TRUE_W * x + TRUE_B + gauss(rng) * 0.18 });
    }
  })();
  let w = 0, b = 0, it = 0, raf = null;

  const lossAt = (w, b) => mseLoss(data, w, b);
  function stepOnce() {
    const eta = n('gd-l', 50) / 1000; // 0.001 .. 0.12
    ({ w, b } = gradientDescentStep(data, w, b, eta)); it++;
    // guard against divergence blowing up the canvas
    if (!Number.isFinite(w) || Math.abs(w) > 1e6) { w = 0; b = 0; }
  }
  function draw() {
    const { ctx, w: W, h: H } = fitCanvas(cv);
    ctx.clearRect(0, 0, W, H);
    setText('gd-lv', (n('gd-l', 50) / 1000).toFixed(3));
    const pad = 30, x0 = pad, x1 = W - 12, y0 = 12, y1 = H - pad;
    const X = x => x0 + (x + 1) / 2 * (x1 - x0);
    const Y = y => y1 - (y + 2) / 4 * (y1 - y0);
    ctx.strokeStyle = RULE; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    // true line (faint)
    ctx.strokeStyle = RULE_H; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(-1), Y(TRUE_W * -1 + TRUE_B)); ctx.lineTo(X(1), Y(TRUE_W * 1 + TRUE_B)); ctx.stroke();
    ctx.setLineDash([]);
    // points
    data.forEach(p => { ctx.fillStyle = 'rgba(67,56,202,0.55)'; ctx.beginPath(); ctx.arc(X(p.x), Y(clamp(p.y, -2, 2)), 3.2, 0, 7); ctx.fill(); });
    // current fit
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(X(-1), Y(clamp(w * -1 + b, -2, 2))); ctx.lineTo(X(1), Y(clamp(w * 1 + b, -2, 2))); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.fillText('dashed = true line', x0 + 6, y0 + 14);

    const loss = lossAt(w, b);
    setText('gd-it', it);
    setText('gd-wb', `${w.toFixed(3)}, ${b.toFixed(3)}`);
    setText('gd-loss', loss.toFixed(4));
    $('gd-loss').style.color = loss < 0.05 ? GOOD : loss > 1 ? BAD : ACCENT;
  }
  function play() {
    if (raf) { cancelAnimationFrame(raf); raf = null; setText('gd-play', 'play'); return; }
    setText('gd-play', 'pause');
    const tick = () => { for (let i = 0; i < 2; i++) stepOnce(); draw(); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  }
  function reset() { if (raf) { cancelAnimationFrame(raf); raf = null; } setText('gd-play', 'play'); w = 0; b = 0; it = 0; draw(); }
  $('gd-play').addEventListener('click', play);
  $('gd-step').addEventListener('click', () => { stepOnce(); draw(); });
  $('gd-reset').addEventListener('click', reset);
  $('gd-l').addEventListener('input', draw);
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 3. BIAS–VARIANCE TRADEOFF — schematic decomposition curve
// ============================================================
(function biasVariance() {
  const cv = $('cv-bv'); if (!cv) return;
  // simple monotone models: bias^2 decays, variance grows with complexity
  const MAXC = 20;
  function curves() {
    const noise = n('bv-n', 15) / 100;
    const bias = c => 0.9 * Math.exp(-0.35 * c);          // decays
    const varc = c => 0.02 * c * c / MAXC;                 // grows ~ quadratically
    return { bias, varc, noise };
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const C = n('bv-c', 5);
    setText('bv-cv', C);
    setText('bv-nv', (n('bv-n', 15) / 100).toFixed(2));
    const { bias, varc, noise } = curves();

    const pad = 34, x0 = pad, x1 = w - 12, y0 = 14, y1 = h - pad;
    const X = c => x0 + (c - 1) / (MAXC - 1) * (x1 - x0);
    const maxY = 1.1;
    const Y = v => y1 - clamp(v, 0, maxY) / maxY * (y1 - y0);
    ctx.strokeStyle = RULE; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    const plot = (fn, col, dash) => {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash || []);
      ctx.beginPath();
      for (let c = 1; c <= MAXC; c += 0.25) { const px = X(c), py = Y(fn(c)); c === 1 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
      ctx.stroke(); ctx.setLineDash([]);
    };
    const total = c => bias(c) ** 2 + varc(c) + noise;
    // find optimum
    let optC = 1, optV = Infinity;
    for (let c = 1; c <= MAXC; c += 0.1) { const t = total(c); if (t < optV) { optV = t; optC = c; } }

    plot(c => bias(c) ** 2, WARN, [5, 4]);
    plot(varc, BAD, [5, 4]);
    plot(total, ACCENT);
    // optimum marker
    ctx.strokeStyle = GOOD; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(optC), y0); ctx.lineTo(X(optC), y1); ctx.stroke(); ctx.setLineDash([]);
    // selected complexity marker
    ctx.fillStyle = ACCENT; ctx.beginPath(); ctx.arc(X(C), Y(total(C)), 4.5, 0, 7); ctx.fill();

    ctx.font = '11px Inter'; ctx.textAlign = 'left';
    ctx.fillStyle = WARN; ctx.fillText('bias²', x1 - 110, y0 + 14);
    ctx.fillStyle = BAD;  ctx.fillText('variance', x1 - 70, y0 + 14);
    ctx.fillStyle = ACCENT; ctx.fillText('total', x1 - 110, y0 + 30);
    ctx.fillStyle = MUTED; ctx.fillText('model complexity →', x0 + 6, y1 + 22);

    setText('bv-bias', (bias(C) ** 2).toFixed(3));
    setText('bv-var', varc(C).toFixed(3));
    setText('bv-tot', total(C).toFixed(3));
    setText('bv-opt', `c ≈ ${optC.toFixed(1)}`);
    $('bv-tot').style.color = Math.abs(C - optC) < 1.5 ? GOOD : ACCENT;
  }
  ['bv-c', 'bv-n'].forEach(id => $(id).addEventListener('input', draw));
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 4. FEATURE SCALING — standardise / min-max a 2-D cloud
// ============================================================
(function scaling() {
  const cv = $('cv-scale'); if (!cv) return;
  let seed = 3, raw = [];
  function regen() {
    const rng = mulberry32(seed);
    raw = [];
    // a cloud with very different per-axis scale and an offset mean
    for (let i = 0; i < 60; i++) {
      raw.push({ x: 50 + gauss(rng) * 18, y: 5 + gauss(rng) * 1.2 });
    }
  }
  const stats = columnStats;
  function transform() {
    const mode = $('sc-m').value;
    if (mode === 'standard') return standardize(raw);
    if (mode === 'minmax') return minMax(raw);
    return raw.map(p => ({ x: p.x, y: p.y }));
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    if (!raw.length) regen();
    const data = transform();
    const sx = stats(data, 'x'), sy = stats(data, 'y');
    // auto-fit viewport with margin
    const pad = 36;
    const xlo = Math.min(...data.map(p => p.x)), xhi = Math.max(...data.map(p => p.x));
    const ylo = Math.min(...data.map(p => p.y)), yhi = Math.max(...data.map(p => p.y));
    const spanx = (xhi - xlo) || 1, spany = (yhi - ylo) || 1;
    const X = x => pad + (x - xlo) / spanx * (w - pad - 16);
    const Y = y => (h - pad) - (y - ylo) / spany * (h - pad - 16);
    // axes through origin if visible
    ctx.strokeStyle = RULE; ctx.lineWidth = 1;
    ctx.strokeRect(pad, 16, w - pad - 16, h - pad - 16);
    if (0 >= xlo && 0 <= xhi) { ctx.strokeStyle = RULE_H; ctx.beginPath(); ctx.moveTo(X(0), 16); ctx.lineTo(X(0), h - pad); ctx.stroke(); }
    if (0 >= ylo && 0 <= yhi) { ctx.strokeStyle = RULE_H; ctx.beginPath(); ctx.moveTo(pad, Y(0)); ctx.lineTo(w - 16, Y(0)); ctx.stroke(); }
    data.forEach(p => { ctx.fillStyle = 'rgba(67,56,202,0.6)'; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 3.4, 0, 7); ctx.fill(); });
    // centroid
    ctx.fillStyle = BAD; ctx.beginPath(); ctx.arc(X(sx.m), Y(sy.m), 5, 0, 7); ctx.fill();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.fillText('red = centroid (mean)', pad + 6, 30);

    setText('sc-mean', `${sx.m.toFixed(2)}, ${sy.m.toFixed(2)}`);
    setText('sc-std', `${sx.sd.toFixed(2)}, ${sy.sd.toFixed(2)}`);
    setText('sc-rx', `[${sx.lo.toFixed(2)}, ${sx.hi.toFixed(2)}]`);
    setText('sc-ry', `[${sy.lo.toFixed(2)}, ${sy.hi.toFixed(2)}]`);
  }
  $('sc-m').addEventListener('change', draw);
  $('sc-new').addEventListener('click', () => { seed = (seed * 22695477 + 1) & 0x7fffffff; regen(); draw(); });
  window.addEventListener('resize', draw);
  regen(); draw();
})();

// ============================================================
// 5. k-NN — decision regions + clickable query
// ============================================================
(function knn() {
  const cv = $('cv-knn'); if (!cv) return;
  let seed = 5, train = [], query = null;
  function regen() {
    const rng = mulberry32(seed);
    train = [];
    // two gaussian blobs in [0,1]^2
    for (let i = 0; i < 24; i++) train.push({ x: 0.33 + gauss(rng) * 0.12, y: 0.40 + gauss(rng) * 0.12, c: 0 });
    for (let i = 0; i < 24; i++) train.push({ x: 0.67 + gauss(rng) * 0.12, y: 0.62 + gauss(rng) * 0.12, c: 1 });
    train.forEach(p => { p.x = clamp(p.x, 0.02, 0.98); p.y = clamp(p.y, 0.02, 0.98); });
    query = null;
  }
  const classify = (x, y, k) => knnClassify(train, x, y, k);
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const k = n('kn-k', 5);
    setText('kn-kv', k);
    if (!train.length) regen();
    const S = Math.min(w, h) - 8, ox = (w - S) / 2, oy = 4;
    const X = x => ox + x * S, Y = y => oy + (1 - y) * S;
    // background grid of class regions
    const step = 14;
    for (let px = 0; px < S; px += step) {
      for (let py = 0; py < S; py += step) {
        const gx = px / S, gy = 1 - py / S;
        const { cls } = classify(gx, gy, k);
        ctx.fillStyle = cls === 0 ? 'rgba(67,56,202,0.10)' : 'rgba(245,158,11,0.12)';
        ctx.fillRect(ox + px, oy + py, step, step);
      }
    }
    ctx.strokeStyle = RULE; ctx.strokeRect(ox, oy, S, S);
    // points
    train.forEach(p => {
      ctx.fillStyle = p.c === 0 ? ACCENT : WARN;
      ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 4, 0, 7); ctx.fill();
    });
    // query
    let res = null;
    if (query) {
      res = classify(query.x, query.y, k);
      // link to k neighbours
      const d = train.map((p, i) => ({ i, d: (p.x - query.x) ** 2 + (p.y - query.y) ** 2 }));
      d.sort((a, b) => a.d - b.d);
      ctx.strokeStyle = RULE_H; ctx.lineWidth = 1;
      for (let i = 0; i < Math.min(k, d.length); i++) {
        const p = train[d[i].i];
        ctx.beginPath(); ctx.moveTo(X(query.x), Y(query.y)); ctx.lineTo(X(p.x), Y(p.y)); ctx.stroke();
      }
      ctx.fillStyle = res.cls === 0 ? ACCENT : WARN;
      ctx.strokeStyle = INK; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(query.x), Y(query.y), 7, 0, 7); ctx.fill(); ctx.stroke();
    }
    // train accuracy (leave-one-out: each point classified by its k nearest others)
    setText('kn-acc', (100 * looAccuracy(train, k)).toFixed(0) + '%');
    if (res) {
      setText('kn-cls', res.cls === 0 ? 'class A' : 'class B');
      $('kn-cls').style.color = res.cls === 0 ? ACCENT : WARN;
      setText('kn-votes', `${res.v0} A · ${res.v1} B`);
    } else { setText('kn-cls', '—'); setText('kn-votes', 'click to query'); $('kn-cls').style.color = INK_S; }
  }
  cv.addEventListener('click', ev => {
    const { w, h } = fitCanvas(cv);
    const S = Math.min(w, h) - 8, ox = (w - S) / 2, oy = 4;
    const p = ptr(cv, ev);
    const x = (p.x - ox) / S, y = 1 - (p.y - oy) / S;
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) { query = { x, y }; draw(); }
  });
  $('kn-k').addEventListener('input', draw);
  $('kn-new').addEventListener('click', () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; regen(); draw(); });
  window.addEventListener('resize', draw);
  regen(); draw();
})();

// ============================================================
// 6. LOGISTIC REGRESSION — gradient descent on cross-entropy
// ============================================================
(function logistic() {
  const cv = $('cv-logit'); if (!cv) return;
  const data = [];
  (function build() {
    const rng = mulberry32(9);
    for (let i = 0; i < 30; i++) data.push({ x: 0.30 + gauss(rng) * 0.13, y: 0.40 + gauss(rng) * 0.13, t: 0 });
    for (let i = 0; i < 30; i++) data.push({ x: 0.70 + gauss(rng) * 0.13, y: 0.62 + gauss(rng) * 0.13, t: 1 });
    data.forEach(p => { p.x = clamp(p.x, 0.02, 0.98); p.y = clamp(p.y, 0.02, 0.98); });
  })();
  let w1 = 0, w2 = 0, b = 0, ep = 0, raf = null;
  const sig = sigmoid;
  function stepEpoch() {
    const eta = n('lg-l', 30) / 100;
    ({ w1, w2, b } = logisticStep({ w1, w2, b }, data, eta)); ep++;
  }
  const metrics = () => logisticMetrics({ w1, w2, b }, data);
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    setText('lg-lv', (n('lg-l', 30) / 100).toFixed(2));
    const S = Math.min(w, h) - 8, ox = (w - S) / 2, oy = 4;
    const X = x => ox + x * S, Y = y => oy + (1 - y) * S;
    // probability shading
    const step = 14;
    for (let px = 0; px < S; px += step) for (let py = 0; py < S; py += step) {
      const gx = px / S, gy = 1 - py / S;
      const pr = sig(w1 * gx + w2 * gy + b);
      const a = (0.18 * Math.abs(pr - 0.5) * 2).toFixed(3);
      ctx.fillStyle = pr >= 0.5 ? `rgba(245,158,11,${a})` : `rgba(67,56,202,${a})`;
      ctx.fillRect(ox + px, oy + py, step, step);
    }
    ctx.strokeStyle = RULE; ctx.strokeRect(ox, oy, S, S);
    // decision boundary w1 x + w2 y + b = 0  ->  y = -(w1 x + b)/w2
    if (Math.abs(w2) > 1e-6) {
      ctx.strokeStyle = INK; ctx.lineWidth = 2;
      const yL = -(w1 * 0 + b) / w2, yR = -(w1 * 1 + b) / w2;
      ctx.beginPath(); ctx.moveTo(X(0), Y(clamp(yL, -0.5, 1.5))); ctx.lineTo(X(1), Y(clamp(yR, -0.5, 1.5))); ctx.stroke();
    }
    data.forEach(p => { ctx.fillStyle = p.t === 0 ? ACCENT : WARN; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 4, 0, 7); ctx.fill(); });

    const { loss, acc } = metrics();
    setText('lg-ep', ep);
    setText('lg-loss', loss.toFixed(4));
    setText('lg-acc', (100 * acc).toFixed(0) + '%');
    $('lg-loss').style.color = loss < 0.3 ? GOOD : loss > 0.65 ? BAD : ACCENT;
  }
  function play() {
    if (raf) { cancelAnimationFrame(raf); raf = null; setText('lg-play', 'play'); return; }
    setText('lg-play', 'pause');
    const tick = () => { stepEpoch(); draw(); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  }
  function reset() { if (raf) { cancelAnimationFrame(raf); raf = null; } setText('lg-play', 'play'); w1 = 0; w2 = 0; b = 0; ep = 0; draw(); }
  $('lg-play').addEventListener('click', play);
  $('lg-step').addEventListener('click', () => { stepEpoch(); draw(); });
  $('lg-reset').addEventListener('click', reset);
  $('lg-l').addEventListener('input', draw);
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 7. DECISION TREE — greedy Gini splits, axis-aligned boxes
// ============================================================
(function decisionTree() {
  const cv = $('cv-tree'); if (!cv) return;
  let seed = 4, data = [];
  function regen() {
    const rng = mulberry32(seed);
    data = [];
    // XOR-ish pattern so depth matters
    for (let i = 0; i < 80; i++) {
      const x = rng(), y = rng();
      const c = ((x > 0.5) !== (y > 0.5)) ? 1 : 0;
      const flip = rng() < 0.08 ? 1 : 0; // a little noise
      data.push({ x, y, c: c ^ flip });
    }
  }
  // gini, bestSplit and the recursive leaf builder come from ./tree.js
  const build = buildTree;
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const maxD = n('tr-d', 3);
    setText('tr-dv', maxD);
    if (!data.length) regen();
    const S = Math.min(w, h) - 8, ox = (w - S) / 2, oy = 4;
    const X = x => ox + x * S, Y = y => oy + (1 - y) * S;
    const leaves = [];
    build(data, { x0: 0, x1: 1, y0: 0, y1: 1 }, 0, maxD, leaves);
    // shade leaves
    leaves.forEach(L => {
      ctx.fillStyle = L.cls === 0 ? 'rgba(67,56,202,0.10)' : 'rgba(245,158,11,0.13)';
      ctx.fillRect(X(L.x0), Y(L.y1), (L.x1 - L.x0) * S, (L.y1 - L.y0) * S);
      ctx.strokeStyle = RULE_H; ctx.lineWidth = 1;
      ctx.strokeRect(X(L.x0), Y(L.y1), (L.x1 - L.x0) * S, (L.y1 - L.y0) * S);
    });
    ctx.strokeStyle = RULE; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, S, S);
    data.forEach(p => { ctx.fillStyle = p.c === 0 ? ACCENT : WARN; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 3.4, 0, 7); ctx.fill(); });

    // train accuracy
    let correct = 0;
    leaves.forEach(L => { correct += L.pts.filter(p => p.c === L.cls).length; });
    setText('tr-leaf', leaves.length);
    setText('tr-rg', gini(data).toFixed(3));
    setText('tr-acc', (100 * correct / data.length).toFixed(0) + '%');
  }
  $('tr-d').addEventListener('input', draw);
  $('tr-new').addEventListener('click', () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; regen(); draw(); });
  window.addEventListener('resize', draw);
  regen(); draw();
})();

// ============================================================
// 8. k-MEANS — Lloyd's algorithm, animated
// ============================================================
(function kmeans() {
  const cv = $('cv-km'); if (!cv) return;
  let seed = 6, pts = [], cents = [], assign = [], it = 0, raf = null, moved = 0;
  const PAL = [ACCENT, WARN, GOOD, BAD, '#0EA5E9', '#9333EA'];
  function regen() {
    const rng = mulberry32(seed);
    pts = [];
    const blobs = 3;
    for (let bI = 0; bI < blobs; bI++) {
      const cx = 0.2 + rng() * 0.6, cy = 0.2 + rng() * 0.6;
      for (let i = 0; i < 25; i++) pts.push({ x: clamp(cx + gauss(rng) * 0.08, 0, 1), y: clamp(cy + gauss(rng) * 0.08, 0, 1) });
    }
    initCents();
  }
  function initCents() {
    const k = n('km-k', 3);
    const rng = mulberry32(seed + 100 + it);
    cents = [];
    for (let i = 0; i < k; i++) cents.push({ x: rng(), y: rng() });
    assign = pts.map(() => -1); it = 0; moved = 0;
  }
  const inertia = () => kmInertia(pts, cents, assign);
  function stepOnce() {
    assign = assignStep(pts, cents);          // assign each point to nearest centroid
    const res = updateStep(pts, cents, assign); // move centroids to member means
    cents = res.cents; moved = res.moved;
    it++;
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    setText('km-kv', n('km-k', 3));
    if (!pts.length) regen();
    if (cents.length !== n('km-k', 3)) initCents();
    const S = Math.min(w, h) - 8, ox = (w - S) / 2, oy = 4;
    const X = x => ox + x * S, Y = y => oy + (1 - y) * S;
    ctx.strokeStyle = RULE; ctx.strokeRect(ox, oy, S, S);
    pts.forEach((p, i) => {
      const col = assign[i] >= 0 ? PAL[assign[i] % PAL.length] : MUTED;
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 3.4, 0, 7); ctx.fill();
    });
    cents.forEach((c, j) => {
      ctx.fillStyle = PAL[j % PAL.length]; ctx.strokeStyle = INK; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.rect(X(c.x) - 6, Y(c.y) - 6, 12, 12); ctx.fill(); ctx.stroke();
    });
    setText('km-it', it);
    setText('km-in', inertia().toFixed(4));
    setText('km-mv', moved.toFixed(4));
    $('km-in').style.color = ACCENT;
  }
  function play() {
    if (raf) { cancelAnimationFrame(raf); raf = null; setText('km-play', 'play'); return; }
    setText('km-play', 'pause');
    let frame = 0;
    const tick = () => {
      frame++;
      if (frame % 20 === 0) { stepOnce(); if (moved < 1e-4 && it > 1) { cancelAnimationFrame(raf); raf = null; setText('km-play', 'play'); draw(); return; } }
      draw(); raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }
  function reset() { if (raf) { cancelAnimationFrame(raf); raf = null; } setText('km-play', 'play'); initCents(); draw(); }
  $('km-play').addEventListener('click', play);
  $('km-step').addEventListener('click', () => { stepOnce(); draw(); });
  $('km-reset').addEventListener('click', reset);
  $('km-new').addEventListener('click', () => { if (raf) { cancelAnimationFrame(raf); raf = null; setText('km-play', 'play'); } seed = (seed * 1103515245 + 12345) & 0x7fffffff; regen(); draw(); });
  $('km-k').addEventListener('input', () => { initCents(); draw(); });
  window.addEventListener('resize', draw);
  regen(); draw();
})();

// ============================================================
// 9. PCA — covariance eigen-decomposition of a 2-D cloud
// ============================================================
(function pca() {
  const cv = $('cv-pca'); if (!cv) return;
  let seed = 8, raw = [];
  function regen() {
    const rng = mulberry32(seed);
    raw = [];
    const r = n('pc-r', 80) / 100;
    for (let i = 0; i < 80; i++) {
      const a = gauss(rng), b = gauss(rng);
      // correlated via Cholesky-like mix
      const x = a;
      const y = r * a + Math.sqrt(Math.max(0, 1 - r * r)) * b;
      raw.push({ x: x * 0.9, y: y * 0.55 });
    }
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    setText('pc-rv', (n('pc-r', 80) / 100).toFixed(2));
    if (!raw.length) regen();
    // covariance eigen-decomposition from ./pca.js
    const { mx, my, l1, l2, v1, v2, kept } = pcaFit(raw);

    const cx = w / 2, cy = h / 2, sc = Math.min(w, h) * 0.32 / 1.2;
    const X = x => cx + (x - mx) * sc, Y = y => cy - (y - my) * sc;
    ctx.strokeStyle = RULE; ctx.strokeRect(8, 8, w - 16, h - 16);
    // projection segments onto PC1
    if ($('pc-proj').checked) {
      ctx.strokeStyle = RULE_H; ctx.lineWidth = 1;
      raw.forEach(p => {
        const dx = p.x - mx, dy = p.y - my;
        const t = dx * v1.x + dy * v1.y;
        const projx = mx + t * v1.x, projy = my + t * v1.y;
        ctx.beginPath(); ctx.moveTo(X(p.x), Y(p.y)); ctx.lineTo(X(projx), Y(projy)); ctx.stroke();
      });
    }
    raw.forEach(p => { ctx.fillStyle = 'rgba(67,56,202,0.55)'; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 3.2, 0, 7); ctx.fill(); });
    // PC1 (solid) and PC2 (dashed), length ~ sqrt(eigenvalue)
    const drawAxis = (vec, lam, col, dash) => {
      const len = Math.sqrt(Math.max(lam, 1e-4)) * 2.2;
      ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.setLineDash(dash || []);
      ctx.beginPath();
      ctx.moveTo(X(mx - vec.x * len), Y(my - vec.y * len));
      ctx.lineTo(X(mx + vec.x * len), Y(my + vec.y * len));
      ctx.stroke(); ctx.setLineDash([]);
    };
    drawAxis(v1, l1, ACCENT);
    drawAxis(v2, l2, WARN, [5, 4]);
    ctx.fillStyle = MUTED; ctx.font = '11px Inter';
    ctx.fillText('PC1 (solid) · PC2 (dashed)', 14, 24);

    const ang = Math.atan2(v1.y, v1.x) * 180 / Math.PI;
    setText('pc-eig', `${l1.toFixed(3)}, ${l2.toFixed(3)}`);
    setText('pc-ang', ang.toFixed(1) + '°');
    setText('pc-var', (100 * kept).toFixed(1) + '%');
  }
  $('pc-r').addEventListener('input', () => { regen(); draw(); });
  $('pc-proj').addEventListener('change', draw);
  $('pc-new').addEventListener('click', () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; regen(); draw(); });
  window.addEventListener('resize', draw);
  regen(); draw();
})();

// ============================================================
// 10. CONFUSION MATRIX — threshold over scored examples
// ============================================================
(function confusion() {
  const cv = $('cv-cm'); if (!cv) return;
  const ex = [];
  (function build() {
    const rng = mulberry32(13);
    // positives score higher on average than negatives, with overlap
    for (let i = 0; i < 60; i++) ex.push({ s: clamp(0.62 + gauss(rng) * 0.18, 0, 1), y: 1 });
    for (let i = 0; i < 60; i++) ex.push({ s: clamp(0.38 + gauss(rng) * 0.18, 0, 1), y: 0 });
  })();
  const counts = thr => confusionFromScores(ex, thr);
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const thr = n('cm-t', 50) / 100;
    setText('cm-tv', thr.toFixed(2));
    const { tp, fp, fn, tn } = counts(thr);

    // left: score strip with threshold; right: 2x2 matrix
    const stripX = 30, stripW = w * 0.42, sy = 30, sh = h - 70;
    ctx.strokeStyle = RULE; ctx.strokeRect(stripX, sy, stripW, sh);
    ctx.fillStyle = MUTED; ctx.font = '11px Inter'; ctx.textAlign = 'left';
    ctx.fillText('score → (1 top)', stripX, sy - 10);
    ex.forEach((e, i) => {
      const px = stripX + (i % 2 === 0 ? stripW * 0.32 : stripW * 0.68);
      const py = sy + (1 - e.s) * sh;
      ctx.fillStyle = e.y === 1 ? WARN : ACCENT;
      ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill();
    });
    const ty = sy + (1 - thr) * sh;
    ctx.strokeStyle = BAD; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(stripX, ty); ctx.lineTo(stripX + stripW, ty); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = BAD; ctx.fillText('threshold', stripX + stripW - 60, ty - 4);
    ctx.fillStyle = ACCENT; ctx.fillText('● actual −', stripX, h - 22);
    ctx.fillStyle = WARN; ctx.fillText('● actual +', stripX + 70, h - 22);

    // matrix
    const mx = w * 0.55, my = 44, cell = Math.min(72, (w - mx - 16) / 2, (h - 70) / 2);
    const grid = [[tp, fn], [fp, tn]];
    const cols = [[GOOD, WARN], [BAD, GOOD]];
    ctx.textAlign = 'center';
    for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) {
      const x = mx + c * cell, y = my + r * cell;
      ctx.fillStyle = (r === c) ? 'rgba(22,163,74,0.12)' : 'rgba(220,38,38,0.10)';
      ctx.fillRect(x, y, cell - 2, cell - 2);
      ctx.strokeStyle = RULE_H; ctx.strokeRect(x, y, cell - 2, cell - 2);
      ctx.fillStyle = cols[r][c]; ctx.font = '700 18px JetBrains Mono, monospace';
      ctx.fillText(grid[r][c], x + cell / 2, y + cell / 2 - 2);
      ctx.fillStyle = MUTED; ctx.font = '10px JetBrains Mono, monospace';
      const lbl = [['TP', 'FN'], ['FP', 'TN']][r][c];
      ctx.fillText(lbl, x + cell / 2, y + cell / 2 + 16);
    }
    ctx.fillStyle = INK_S; ctx.font = '11px Inter';
    ctx.fillText('pred +    pred −', mx + cell, my - 8);
    ctx.save(); ctx.translate(mx - 12, my + cell); ctx.rotate(-Math.PI / 2);
    ctx.fillText('actual −   actual +', 0, 0); ctx.restore();
    ctx.textAlign = 'left';

    const cm = { tp, fp, fn, tn };
    const acc = cmAccuracy(cm), prec = cmPrecision(cm), rec = cmRecall(cm), f1 = cmF1(cm);
    setText('cm-cnt', `${tp} · ${fp} · ${fn} · ${tn}`);
    setText('cm-acc', acc.toFixed(3));
    setText('cm-prec', prec.toFixed(3));
    setText('cm-rec', rec.toFixed(3));
    setText('cm-f1', f1.toFixed(3));
    $('cm-f1').style.color = f1 > 0.8 ? GOOD : f1 < 0.5 ? BAD : ACCENT;
  }
  $('cm-t').addEventListener('input', draw);
  window.addEventListener('resize', draw);
  draw();
})();

// ============================================================
// 11. ROC / AUC — sweep threshold over two scored populations
// ============================================================
(function roc() {
  const cv = $('cv-roc'); if (!cv) return;
  let seed = 17, pos = [], neg = [];
  function regen() {
    const rng = mulberry32(seed);
    const sep = n('rc-s', 15) / 10; // 0 .. 4 standard-deviations of separation
    pos = []; neg = [];
    for (let i = 0; i < 80; i++) pos.push(clamp(0.5 + sep * 0.12 + gauss(rng) * 0.14, 0, 1));
    for (let i = 0; i < 80; i++) neg.push(clamp(0.5 - sep * 0.12 + gauss(rng) * 0.14, 0, 1));
  }
  function rates(thr) {
    const tp = pos.filter(s => s >= thr).length, fn = pos.length - tp;
    const fp = neg.filter(s => s >= thr).length, tn = neg.length - fp;
    return { tpr: tp / (tp + fn || 1), fpr: fp / (fp + tn || 1) };
  }
  const rocCurve = () => rocAuc(pos, neg); // ROC sweep + trapezoidal AUC from ./metrics.js
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    setText('rc-sv', (n('rc-s', 15) / 10).toFixed(1));
    const thr = n('rc-t', 50) / 100;
    setText('rc-tv', thr.toFixed(2));
    if (!pos.length) regen();

    const S = Math.min(w, h) - 40, ox = 34, oy = 8;
    const X = x => ox + x * S, Y = y => oy + (1 - y) * S;
    ctx.strokeStyle = RULE; ctx.strokeRect(ox, oy, S, S);
    // chance diagonal
    ctx.strokeStyle = RULE_H; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.setLineDash([]);
    const { pts, auc } = rocCurve();
    // filled area
    ctx.fillStyle = ACCENT_S; ctx.beginPath(); ctx.moveTo(X(0), Y(0));
    pts.forEach(p => ctx.lineTo(X(p.fpr), Y(p.tpr)));
    ctx.lineTo(X(1), Y(0)); ctx.closePath(); ctx.fill();
    // curve
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2.2; ctx.beginPath();
    pts.forEach((p, i) => { const px = X(p.fpr), py = Y(p.tpr); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); });
    ctx.stroke();
    // current operating point
    const { tpr, fpr } = rates(thr);
    ctx.fillStyle = BAD; ctx.beginPath(); ctx.arc(X(fpr), Y(tpr), 5, 0, 7); ctx.fill();
    ctx.fillStyle = MUTED; ctx.font = '11px Inter';
    ctx.fillText('FPR →', ox + S / 2 - 16, oy + S + 24);
    ctx.save(); ctx.translate(ox - 18, oy + S / 2 + 16); ctx.rotate(-Math.PI / 2); ctx.fillText('TPR →', 0, 0); ctx.restore();

    setText('rc-tpr', tpr.toFixed(3));
    setText('rc-fpr', fpr.toFixed(3));
    setText('rc-auc', auc.toFixed(3));
    $('rc-auc').style.color = auc > 0.85 ? GOOD : auc < 0.6 ? BAD : ACCENT;
  }
  $('rc-s').addEventListener('input', () => { regen(); draw(); });
  $('rc-t').addEventListener('input', draw);
  $('rc-new').addEventListener('click', () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; regen(); draw(); });
  window.addEventListener('resize', draw);
  regen(); draw();
})();

// ============================================================
// 12. NEURAL NETWORK (MLP) — one hidden tanh layer learns a non-linear
//     (XOR) boundary that the linear logistic model above cannot.
// ============================================================
(function neuralNet() {
  const cv = $('cv-mlp'); if (!cv) return;
  let seed = 11, data = [], net = null, ep = 0, raf = null;

  // XOR-style data: classes occupy opposite quadrants, so no straight line
  // separates them — the hidden layer is what makes it learnable.
  function regen() {
    const rng = mulberry32(seed);
    data = [];
    const blob = (cx, cy, t) => { for (let i = 0; i < 22; i++)
      data.push({ x: clamp(cx + gauss(rng) * 0.07, 0.02, 0.98), y: clamp(cy + gauss(rng) * 0.07, 0.02, 0.98), t }); };
    blob(0.27, 0.27, 0); blob(0.73, 0.27, 1); blob(0.27, 0.73, 1); blob(0.73, 0.73, 0);
  }
  function rebuild() {
    if (raf) { cancelAnimationFrame(raf); raf = null; setText('nn-play', 'play'); }
    net = initMLP(Math.round(n('nn-h', 8)), seed); ep = 0;
  }
  function stepEpoch() { net = trainStep(net, data, n('nn-l', 60) / 100); ep++; }

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    setText('nn-hv', Math.round(n('nn-h', 8)));
    setText('nn-lv', (n('nn-l', 60) / 100).toFixed(2));
    const S = Math.min(w, h) - 8, ox = (w - S) / 2, oy = 4;
    const X = x => ox + x * S, Y = y => oy + (1 - y) * S;
    // probability shading — the curved decision surface the network has learned
    const step = 12;
    for (let px = 0; px < S; px += step) for (let py = 0; py < S; py += step) {
      const gx = px / S, gy = 1 - py / S;
      const pr = forward(net, { x: gx, y: gy }).p;
      const a = (0.20 * Math.abs(pr - 0.5) * 2).toFixed(3);
      ctx.fillStyle = pr >= 0.5 ? `rgba(245,158,11,${a})` : `rgba(67,56,202,${a})`;
      ctx.fillRect(ox + px, oy + py, step, step);
    }
    ctx.strokeStyle = RULE; ctx.strokeRect(ox, oy, S, S);
    data.forEach(p => { ctx.fillStyle = p.t === 0 ? ACCENT : WARN; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 4, 0, 7); ctx.fill(); });

    const { loss, acc } = mlpMetrics(net, data);
    setText('nn-ep', ep);
    setText('nn-loss', loss.toFixed(4));
    setText('nn-acc', (100 * acc).toFixed(0) + '%');
    $('nn-acc').style.color = acc > 0.9 ? GOOD : acc < 0.7 ? BAD : ACCENT;
  }
  function play() {
    if (raf) { cancelAnimationFrame(raf); raf = null; setText('nn-play', 'play'); return; }
    setText('nn-play', 'pause');
    const tick = () => { for (let i = 0; i < 3; i++) stepEpoch(); draw(); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  }
  $('nn-play').addEventListener('click', play);
  $('nn-step').addEventListener('click', () => { stepEpoch(); draw(); });
  $('nn-reset').addEventListener('click', () => { rebuild(); draw(); });
  $('nn-h').addEventListener('input', () => { rebuild(); draw(); });
  $('nn-l').addEventListener('input', draw);
  $('nn-new').addEventListener('click', () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; regen(); rebuild(); draw(); });
  window.addEventListener('resize', draw);
  regen(); rebuild(); draw();
})();

// ============================================================
// 13. K-FOLD CROSS-VALIDATION — rotate which fold is held out, average the
//     validation errors (Model Evaluation, session 13). A lower-variance,
//     less optimistic estimate of generalization than one train/test split.
// ============================================================
(function crossVal() {
  const cv = $('cv-kf'); if (!cv) return;
  let seed = 5, pts = [], foldI = 0;
  const truth = x => Math.sin(2 * Math.PI * x) * 0.6 + 0.05;
  function regen() {
    const rng = mulberry32(seed);
    pts = [];
    for (let i = 0; i < 30; i++) { const x = rng(); pts.push({ x, y: truth(x) + gauss(rng) * 0.22 }); }
    foldI = 0;
  }
  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const deg = Math.round(n('kf-d', 4)), k = Math.round(n('kf-k', 5));
    setText('kf-dv', deg); setText('kf-kv', k);
    if (!pts.length) regen();
    const out = crossValidate(pts, { degree: deg, k, seed });
    foldI = ((foldI % out.k) + out.k) % out.k;
    const sel = out.results[foldI];
    const valSet = new Set(sel.valIdx);

    // plot frame (top region)
    const pad = 38, x0 = pad, x1 = w - 14, y0 = 14, y1 = h - 116;
    const X = x => x0 + x * (x1 - x0);
    const Y = y => y1 - (y + 1.3) / 2.6 * (y1 - y0);
    ctx.strokeStyle = RULE; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    // fitted curve on the training folds
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= 220; i++) { const x = i / 220, y = clamp(evalPoly(sel.coef, x), -1.3, 1.3); const px = X(x), py = Y(y); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
    ctx.stroke();
    // points: training (filled) vs the held-out validation fold (amber rings)
    pts.forEach((p, i) => {
      const px = X(p.x), py = Y(clamp(p.y, -1.3, 1.3));
      if (valSet.has(i)) { ctx.strokeStyle = WARN; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 7); ctx.stroke(); }
      else { ctx.fillStyle = 'rgba(67,56,202,0.55)'; ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill(); }
    });
    ctx.fillStyle = MUTED; ctx.font = '11px Inter';
    ctx.fillText(`● train   ○ validation (fold ${foldI + 1})`, x0 + 6, y0 + 14);

    // per-fold validation-RMSE bars (bottom region) with the CV mean line
    const errs = out.results.map(r => r.rmse), emax = Math.max(...errs, 0.05);
    const bx0 = pad, bw = (x1 - x0) / out.k, by1 = h - 26, bh = 64, by0 = by1 - bh;
    ctx.strokeStyle = RULE; ctx.beginPath(); ctx.moveTo(bx0, by1); ctx.lineTo(x1, by1); ctx.stroke();
    errs.forEach((e, i) => {
      const x = bx0 + i * bw + bw * 0.18, bwid = bw * 0.64, hgt = e / emax * bh;
      ctx.fillStyle = i === foldI ? WARN : ACCENT_S;
      ctx.fillRect(x, by1 - hgt, bwid, hgt);
      ctx.strokeStyle = i === foldI ? WARN : ACCENT; ctx.lineWidth = 1; ctx.strokeRect(x, by1 - hgt, bwid, hgt);
    });
    const my = by1 - out.mean / emax * bh;
    ctx.strokeStyle = BAD; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(bx0, my); ctx.lineTo(x1, my); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = BAD; ctx.font = '10px Inter'; ctx.textAlign = 'right';
    ctx.fillText('CV mean', x1 - 2, my - 3); ctx.textAlign = 'left';
    ctx.fillStyle = MUTED; ctx.fillText('per-fold validation RMSE', bx0, by0 - 4);

    setText('kf-fold', `${foldI + 1} / ${out.k}`);
    setText('kf-foldrmse', sel.rmse.toFixed(3));
    setText('kf-cv', `${out.mean.toFixed(3)} ± ${out.std.toFixed(3)}`);
    $('kf-cv').style.color = out.mean < 0.3 ? GOOD : out.mean > 0.55 ? BAD : ACCENT;
  }
  $('kf-d').addEventListener('input', draw);
  $('kf-k').addEventListener('input', () => { foldI = 0; draw(); });
  $('kf-next').addEventListener('click', () => { foldI++; draw(); });
  $('kf-new').addEventListener('click', () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; regen(); draw(); });
  window.addEventListener('resize', draw);
  regen(); draw();
})();
