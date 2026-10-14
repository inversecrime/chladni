// Tests for solver.js against the original Python implementation (reference.json, made by
// make_reference.py). Run: node --test tests/
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const S = require("../solver.js");

const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "reference.json"), "utf8"));

test("operator A and B match the Python matrices entry by entry", () => {
  for (const o of ref.operators) {
    const op = S.biharmonicOperator(o.n, o.mu);
    const nd = o.n * o.n;
    let scale = 0;
    for (const v of o.vals) scale = Math.max(scale, Math.abs(v));
    // every Python entry is reproduced
    const seen = new Set();
    let maxDiff = 0;
    o.rows.forEach((i, k) => {
      maxDiff = Math.max(maxDiff, Math.abs((op.rows[i].get(o.cols[k]) || 0) - o.vals[k]));
      seen.add(i * nd + o.cols[k]);
    });
    // and nothing extra
    for (let i = 0; i < nd; i++) {
      for (const [j, v] of op.rows[i]) if (!seen.has(i * nd + j)) maxDiff = Math.max(maxDiff, Math.abs(v));
    }
    assert.ok(maxDiff < 1e-10 * scale, `n=${o.n} mu=${o.mu}: max entry difference ${maxDiff} (scale ${scale})`);
    o.bdiag.forEach((b, i) => assert.ok(Math.abs(op.bdiag[i] - b) < 1e-14, `bdiag ${i}`));
  }
});

test("operator is symmetric and has three rigid-body modes (A 1 = 0, A x = 0 for x = linear)", () => {
  const n = 15;
  const op = S.biharmonicOperator(n, 0.4);
  const nd = n * n;
  for (let i = 0; i < nd; i++) {
    for (const [j, v] of op.rows[i]) assert.ok(Math.abs(v - (op.rows[j].get(i) || 0)) < 1e-9 * Math.abs(v) + 1e-9);
  }
  const csr = S.toCSR(op.rows);
  const y = new Float64Array(nd);
  for (const f of [() => 1, (r) => r, (r, c) => c]) {
    const x = new Float64Array(nd);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) x[r * n + c] = f(r, c);
    S.csrMul(csr, x, y);
    let big = 0;
    for (const v of y) big = Math.max(big, Math.abs(v));
    assert.ok(big < 1e-8, `rigid mode not in kernel: ${big}`);
  }
});

test("eigenvalues match scipy/ARPACK (relative error < 1e-7)", () => {
  for (const m of ref.modes) {
    const t0 = Date.now();
    const { eigenvalues } = S.calculatePatterns(m.n, m.mu, 1, m.k);
    const ms = Date.now() - t0;
    let worst = 0;
    for (let i = 0; i < m.k; i++) worst = Math.max(worst, Math.abs(eigenvalues[i] - m.eigenvalues[i]) / m.eigenvalues[i]);
    console.log(`  n=${m.n} mu=${m.mu} k=${m.k}: worst relative eigenvalue error ${worst.toExponential(2)}, ${ms} ms`);
    assert.ok(worst < 1e-7, `n=${m.n} mu=${m.mu}: ${worst}`);
  }
});

test("eigenvectors span the same eigenspaces as scipy (degenerate pairs handled)", () => {
  for (const m of ref.modes.filter((e) => e.eigenvectors)) {
    const { eigenvalues, patterns } = S.calculatePatterns(m.n, m.mu, 1, m.k);
    const py = m.eigenvectors.map((v) => {
      const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      return v.map((x) => x / norm);
    });
    // cluster by eigenvalue
    let start = 0;
    while (start < m.k) {
      let end = start + 1;
      while (end < m.k && Math.abs(m.eigenvalues[end] - m.eigenvalues[end - 1]) < 1e-6 * m.eigenvalues[end]) end++;
      if (end === m.k) break; // last cluster could be cut off by k, skip it
      // sum of squared overlaps of mine with the python subspace must equal the cluster size
      let total = 0;
      for (let i = start; i < end; i++) {
        for (let j = start; j < end; j++) {
          let dot = 0;
          for (let q = 0; q < patterns[i].length; q++) dot += patterns[i][q] * py[j][q];
          total += dot * dot;
        }
      }
      assert.ok(Math.abs(total - (end - start)) < 1e-6, `n=${m.n} mu=${m.mu} modes ${start}..${end - 1}: overlap ${total}`);
      start = end;
    }
  }
});

test("eigenpairs satisfy A x = lambda B x on the production grid (n=50)", () => {
  const n = 50;
  const op = S.biharmonicOperator(n, 0.37);
  const { eigenvalues, patterns } = S.calculatePatterns(n, 0.37, 1, 30);
  const csr = S.toCSR(op.rows);
  const y = new Float64Array(n * n);
  patterns.forEach((x, k) => {
    S.csrMul(csr, x, y);
    let r2 = 0;
    let n2 = 0;
    for (let i = 0; i < y.length; i++) {
      const r = y[i] - eigenvalues[k] * op.bdiag[i] * x[i];
      r2 += (r * r) / op.bdiag[i];
      n2 += eigenvalues[k] * eigenvalues[k] * op.bdiag[i] * x[i] * x[i];
    }
    assert.ok(Math.sqrt(r2 / n2) < 1e-6, `mode ${k}: relative residual ${Math.sqrt(r2 / n2)}`);
  });
});

test("rigid-body modes come out as the three zero eigenvalues", () => {
  const op = S.biharmonicOperator(20, 0.5);
  const { pairs } = S.lowestEigenpairs(op, 6);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(pairs[i].value) < 1e-6, `eigenvalue ${i} = ${pairs[i].value}`);
  assert.ok(pairs[3].value > 1);
});

test("calculatePatterns: frequencies, sorting, normalization, sign convention", () => {
  const { eigenvalues, frequencies, patterns } = S.calculatePatterns(30, 0.5, 8, 30);
  assert.equal(patterns.length, 30);
  for (let i = 0; i < 30; i++) {
    assert.ok(eigenvalues[i] > 1);
    if (i > 0) assert.ok(eigenvalues[i] >= eigenvalues[i - 1]);
    assert.ok(Math.abs(frequencies[i] - 8 * Math.sqrt(eigenvalues[i])) < 1e-12);
    const norm = Math.sqrt(patterns[i].reduce((s, x) => s + x * x, 0));
    assert.ok(Math.abs(norm - 1) < 1e-12);
    let big = 0;
    for (const x of patterns[i]) if (Math.abs(x) > Math.abs(big)) big = x;
    assert.ok(big > 0);
  }
});

test("solver is deterministic and independent of the random start", () => {
  const a = S.calculatePatterns(25, 0.6, 1, 20, { seed: 1 });
  const b = S.calculatePatterns(25, 0.6, 1, 20, { seed: 99 });
  for (let i = 0; i < 20; i++) assert.ok(Math.abs(a.eigenvalues[i] - b.eigenvalues[i]) < 1e-7 * a.eigenvalues[i]);
});

test("production grid solves in a reasonable time", () => {
  const t0 = Date.now();
  S.calculatePatterns(50, 0.5, 8, 30);
  const ms = Date.now() - t0;
  console.log(`  n=50, 30 patterns: ${ms} ms`);
  assert.ok(ms < 20000);
});

test("band Cholesky solves a random SPD band system", () => {
  const n = 200;
  const bw = 7;
  const rng = S.makeRng(5);
  const rows = Array.from({ length: n }, () => new Map());
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(0, i - bw); j < i; j++) {
      const v = rng() - 0.5;
      rows[i].set(j, v);
      rows[j].set(i, v);
    }
  }
  for (let i = 0; i < n; i++) rows[i].set(i, 10);
  const csr = S.toCSR(rows);
  const x = Float64Array.from({ length: n }, () => rng());
  const b = new Float64Array(n);
  S.csrMul(csr, x, b);
  const fac = S.bandCholesky(csr, new Float64Array(n), S.bandwidth(csr));
  const out = new Float64Array(n);
  S.bandSolve(fac, b, out);
  for (let i = 0; i < n; i++) assert.ok(Math.abs(out[i] - x[i]) < 1e-12);
});

test("symmetricEigen diagonalizes a symmetric matrix", () => {
  const rng = S.makeRng(3);
  const n = 30;
  const a = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) a[i][j] = a[j][i] = rng() - 0.5;
  const orig = a.map((r) => r.slice());
  const { values, vectors } = S.symmetricEigen(a);
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += orig[i][j] * vectors[k][j];
      assert.ok(Math.abs(s - values[k] * vectors[k][i]) < 1e-10);
    }
  }
});
