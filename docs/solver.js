// Chladni plate eigenmodes in pure JavaScript (no dependencies).
//
// Port of chladni_maths.py: finite-volume / finite-difference discretization of the free square
// plate (Gander & Kwok, "Chladni Figures and the Tacoma Bridge"), then a generalized symmetric
// eigenproblem A x = lambda B x with B diagonal. scipy's ARPACK is replaced by a block Krylov
// method on the shift-inverted operator (A + s B)^-1 B with full reorthogonalization,
// Rayleigh-Ritz extraction and a banded Cholesky factorization.
//
// Works in the browser (global `ChladniSolver`) and in node (`require`).

(function (root) {
  "use strict";

  // ---------------------------------------------------------------- sparse rows

  // A sparse matrix is an array of Maps (row -> Map(col -> value)); only used while assembling.
  function newRows(count) {
    const rows = new Array(count);
    for (let i = 0; i < count; i++) rows[i] = new Map();
    return rows;
  }

  function add(rows, i, j, v) {
    const row = rows[i];
    row.set(j, (row.get(j) || 0) + v);
  }

  // (X @ Y) for sparse row matrices.
  function matmul(X, Y) {
    const out = newRows(X.length);
    for (let i = 0; i < X.length; i++) {
      for (const [k, xv] of X[i]) {
        for (const [j, yv] of Y[k]) add(out, i, j, xv * yv);
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- operator assembly

  // Triplets (dr, dc, value) of the stencils in chladni_maths.one_side_of_L.
  const STENCIL_L = {
    horizontal: {
      plus: [[0, -1, 0.5], [0, 1, -0.5], [1, -1, -0.5], [1, 1, 0.5]],
      minus: [[-1, -1, 0.5], [-1, 1, -0.5], [0, -1, -0.5], [0, 1, 0.5]],
    },
    vertical: {
      plus: [[-1, 0, 0.5], [-1, 1, -0.5], [1, 0, -0.5], [1, 1, 0.5]],
      minus: [[-1, -1, 0.5], [-1, 0, -0.5], [1, -1, -0.5], [1, 0, 0.5]],
    },
  };

  /**
   * Builds the matrices A and B of the eigenvalue problem A x = lambda B x on an n x n grid.
   * Returns { n, rows, bdiag } where `rows[i]` is a Map col -> value (A is symmetric) and
   * `bdiag` the diagonal of B. Mirrors chladni_maths.biharmonic_operator.
   */
  function biharmonicOperator(n, mu) {
    const m = n + 2;
    const f = (r, c) => r * m + c;
    const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
    const inner = range(1, m - 1);

    const edgeLeft = inner.map((r) => f(r, 1));
    const edgeRight = inner.map((r) => f(r, m - 2));
    const edgeTop = inner.map((c) => f(m - 2, c));
    const edgeBottom = inner.map((c) => f(1, c));
    const ghostLeft = inner.map((r) => f(r, 0));
    const ghostRight = inner.map((r) => f(r, m - 1));
    const ghostTop = inner.map((c) => f(m - 1, c));
    const ghostBottom = inner.map((c) => f(0, c));
    const ghost = [].concat(ghostLeft, ghostRight, ghostTop, ghostBottom);

    // Unscaled 5-point laplacian, nonzero rows only for domain points (rows 1..m-2, cols 1..m-2).
    const lap = newRows(m * m);
    for (const r of inner) {
      for (const c of inner) {
        const i = f(r, c);
        add(lap, i, i, -4);
        add(lap, i, f(r - 1, c), 1);
        add(lap, i, f(r + 1, c), 1);
        add(lap, i, f(r, c - 1), 1);
        add(lap, i, f(r, c + 1), 1);
      }
    }

    // N = -lap, with the entries inside each edge set halved (corners get quartered).
    const N = newRows(m * m);
    for (let i = 0; i < m * m; i++) for (const [j, v] of lap[i]) N[i].set(j, -v);
    for (const edge of [edgeLeft, edgeRight, edgeTop, edgeBottom]) {
      const inEdge = new Set(edge);
      for (const i of edge) {
        for (const [j, v] of N[i]) if (inEdge.has(j)) N[i].set(j, v / 2);
      }
    }

    // L = -lap + (sides), the sides carry the boundary integrals in the ghost rows.
    const L = newRows(m * m);
    for (let i = 0; i < m * m; i++) for (const [j, v] of lap[i]) L[i].set(j, -v);

    function addSideOfL(orientation, ghostIdx, boundaryIdx, sign) {
      const st = STENCIL_L[orientation];
      const factor = -(1 - mu) * sign;
      for (let k = 0; k + 1 < ghostIdx.length; k++) {
        for (const [dr, dc, v] of st.plus) add(L, ghostIdx[k], boundaryIdx[k] + dr * m + dc, factor * v);
        for (const [dr, dc, v] of st.minus) add(L, ghostIdx[k + 1], boundaryIdx[k + 1] + dr * m + dc, -factor * v);
      }
    }
    addSideOfL("horizontal", ghostLeft, edgeLeft, +1);
    addSideOfL("horizontal", ghostRight, edgeRight, -1);
    addSideOfL("vertical", ghostTop, edgeTop, -1);
    addSideOfL("vertical", ghostBottom, edgeBottom, +1);

    // A = N L + ghost point equations.
    const A = matmul(N, L);
    function addSideOfA(orientation, ghostIdx, boundaryIdx) {
      const st =
        orientation === "horizontal"
          ? [[-1, 0, -mu], [0, -1, -1], [0, 0, 2 * (1 + mu)], [0, 1, -1], [1, 0, -mu]]
          : [[-1, 0, -1], [0, -1, -mu], [0, 0, 2 * (1 + mu)], [0, 1, -mu], [1, 0, -1]];
      for (let k = 0; k < ghostIdx.length; k++) {
        for (const [dr, dc, v] of st) add(A, ghostIdx[k], boundaryIdx[k] + dr * m + dc, v);
      }
    }
    addSideOfA("horizontal", ghostLeft, edgeLeft);
    addSideOfA("horizontal", ghostRight, edgeRight);
    addSideOfA("vertical", ghostTop, edgeTop);
    addSideOfA("vertical", ghostBottom, edgeBottom);

    // Eliminate the ghost points: A = A0 - A1 A3^-1 A2.
    const domainIndex = new Int32Array(m * m).fill(-1);
    for (let r = 1; r < m - 1; r++) for (let c = 1; c < m - 1; c++) domainIndex[f(r, c)] = (r - 1) * n + (c - 1);
    const ghostIndex = new Int32Array(m * m).fill(-1);
    ghost.forEach((g, k) => (ghostIndex[g] = k));
    const ng = ghost.length;
    const nd = n * n;

    // Dense A3 and its inverse (Gauss-Jordan with partial pivoting).
    const inv = Array.from({ length: ng }, (_, i) => {
      const row = new Float64Array(2 * ng);
      row[ng + i] = 1;
      return row;
    });
    ghost.forEach((g, i) => {
      for (const [j, v] of A[g]) if (ghostIndex[j] >= 0) inv[i][ghostIndex[j]] += v;
    });
    for (let col = 0; col < ng; col++) {
      let piv = col;
      for (let r = col + 1; r < ng; r++) if (Math.abs(inv[r][col]) > Math.abs(inv[piv][col])) piv = r;
      if (Math.abs(inv[piv][col]) < 1e-300) throw new Error("singular ghost block");
      [inv[col], inv[piv]] = [inv[piv], inv[col]];
      const p = inv[col][col];
      for (let j = 0; j < 2 * ng; j++) inv[col][j] /= p;
      for (let r = 0; r < ng; r++) {
        if (r === col) continue;
        const factor = inv[r][col];
        if (factor === 0) continue;
        const rowR = inv[r];
        const rowC = inv[col];
        for (let j = 0; j < 2 * ng; j++) rowR[j] -= factor * rowC[j];
      }
    }

    // W = A3^-1 A2 (ghost x domain, dense).
    const W = Array.from({ length: ng }, () => new Float64Array(nd));
    ghost.forEach((g, k) => {
      for (const [j, v] of A[g]) {
        const dj = domainIndex[j];
        if (dj < 0) continue;
        for (let i = 0; i < ng; i++) {
          const a = inv[i][ng + k];
          if (a !== 0) W[i][dj] += a * v;
        }
      }
    });

    const h = 2 / (n - 1);
    const scale = 1 / (h * h);
    const rows = new Array(nd);
    const dense = new Float64Array(nd);
    for (let r = 1; r < m - 1; r++) {
      for (let c = 1; c < m - 1; c++) {
        dense.fill(0);
        for (const [j, v] of A[f(r, c)]) {
          if (domainIndex[j] >= 0) dense[domainIndex[j]] += v;
          else if (ghostIndex[j] >= 0) {
            const Wg = W[ghostIndex[j]];
            for (let q = 0; q < nd; q++) dense[q] -= v * Wg[q];
          }
        }
        let big = 0;
        for (let q = 0; q < nd; q++) big = Math.max(big, Math.abs(dense[q]));
        const row = new Map();
        for (let q = 0; q < nd; q++) if (Math.abs(dense[q]) > 1e-12 * big) row.set(q, dense[q] * scale);
        rows[domainIndex[f(r, c)]] = row;
      }
    }

    // B: identity, halved on edges (quartered in corners), scaled by h^2.
    const bdiag = new Float64Array(nd);
    for (let r = 1; r < m - 1; r++) {
      for (let c = 1; c < m - 1; c++) {
        let v = 1;
        if (c === 1) v /= 2;
        if (c === m - 2) v /= 2;
        if (r === 1) v /= 2;
        if (r === m - 2) v /= 2;
        bdiag[domainIndex[f(r, c)]] = v * h * h;
      }
    }

    // Symmetrize away round-off (A is symmetric in exact arithmetic).
    for (let i = 0; i < nd; i++) {
      for (const [j, v] of rows[i]) {
        if (j > i) {
          const mean = (v + (rows[j].get(i) || 0)) / 2;
          rows[i].set(j, mean);
          rows[j].set(i, mean);
        }
      }
    }

    return { n, rows, bdiag };
  }

  // ---------------------------------------------------------------- CSR + banded Cholesky

  function toCSR(rows) {
    const nd = rows.length;
    let nnz = 0;
    for (const row of rows) nnz += row.size;
    const ptr = new Int32Array(nd + 1);
    const idx = new Int32Array(nnz);
    const val = new Float64Array(nnz);
    let p = 0;
    for (let i = 0; i < nd; i++) {
      ptr[i] = p;
      for (const [j, v] of rows[i]) {
        idx[p] = j;
        val[p] = v;
        p++;
      }
    }
    ptr[nd] = p;
    return { n: nd, ptr, idx, val };
  }

  function csrMul(csr, x, y) {
    for (let i = 0; i < csr.n; i++) {
      let s = 0;
      for (let p = csr.ptr[i]; p < csr.ptr[i + 1]; p++) s += csr.val[p] * x[csr.idx[p]];
      y[i] = s;
    }
  }

  function bandwidth(csr) {
    let bw = 0;
    for (let i = 0; i < csr.n; i++) {
      for (let p = csr.ptr[i]; p < csr.ptr[i + 1]; p++) bw = Math.max(bw, Math.abs(i - csr.idx[p]));
    }
    return bw;
  }

  // Cholesky K = L L^T of a symmetric positive definite band matrix. L stored row-wise:
  // entry (i, j), i-bw <= j <= i, lives at l[i * (bw + 1) + (j - i + bw)].
  function bandCholesky(csr, shiftDiag, bw) {
    const n = csr.n;
    const w = bw + 1;
    const l = new Float64Array(n * w);
    for (let i = 0; i < n; i++) {
      for (let p = csr.ptr[i]; p < csr.ptr[i + 1]; p++) {
        const j = csr.idx[p];
        if (j <= i) l[i * w + (j - i + bw)] += csr.val[p];
      }
      l[i * w + bw] += shiftDiag[i];
    }
    for (let i = 0; i < n; i++) {
      const jmin = Math.max(0, i - bw);
      for (let j = jmin; j <= i; j++) {
        let s = l[i * w + (j - i + bw)];
        const kmin = Math.max(jmin, j - bw);
        for (let k = kmin; k < j; k++) s -= l[i * w + (k - i + bw)] * l[j * w + (k - j + bw)];
        if (j === i) {
          if (!(s > 0)) throw new Error("matrix not positive definite");
          l[i * w + bw] = Math.sqrt(s);
        } else {
          l[i * w + (j - i + bw)] = s / l[j * w + bw];
        }
      }
    }
    return { n, bw, l };
  }

  function bandSolve(fac, b, x) {
    const { n, bw, l } = fac;
    const w = bw + 1;
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = Math.max(0, i - bw); k < i; k++) s -= l[i * w + (k - i + bw)] * x[k];
      x[i] = s / l[i * w + bw];
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = x[i];
      const kmax = Math.min(n - 1, i + bw);
      for (let k = i + 1; k <= kmax; k++) s -= l[k * w + (i - k + bw)] * x[k];
      x[i] = s / l[i * w + bw];
    }
  }

  // ---------------------------------------------------------------- small dense symmetric eigensolver

  // Householder tridiagonalization + implicit QL (the classic tred2/tql2 pair).
  // `a`: array of Float64Array rows (symmetric, destroyed). Returns { values, vectors } with
  // vectors[k] the eigenvector of values[k] (unsorted).
  function symmetricEigen(a) {
    const n = a.length;
    const V = a;
    const d = new Float64Array(n);
    const e = new Float64Array(n);
    for (let j = 0; j < n; j++) d[j] = V[n - 1][j];

    for (let i = n - 1; i > 0; i--) {
      let scale = 0;
      let h = 0;
      for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
      if (scale === 0) {
        e[i] = d[i - 1];
        for (let j = 0; j < i; j++) {
          d[j] = V[i - 1][j];
          V[i][j] = 0;
          V[j][i] = 0;
        }
      } else {
        for (let k = 0; k < i; k++) {
          d[k] /= scale;
          h += d[k] * d[k];
        }
        let f = d[i - 1];
        let g = Math.sqrt(h);
        if (f > 0) g = -g;
        e[i] = scale * g;
        h -= f * g;
        d[i - 1] = f - g;
        for (let j = 0; j < i; j++) e[j] = 0;
        for (let j = 0; j < i; j++) {
          f = d[j];
          V[j][i] = f;
          g = e[j] + V[j][j] * f;
          for (let k = j + 1; k <= i - 1; k++) {
            g += V[k][j] * d[k];
            e[k] += V[k][j] * f;
          }
          e[j] = g;
        }
        f = 0;
        for (let j = 0; j < i; j++) {
          e[j] /= h;
          f += e[j] * d[j];
        }
        const hh = f / (h + h);
        for (let j = 0; j < i; j++) e[j] -= hh * d[j];
        for (let j = 0; j < i; j++) {
          f = d[j];
          g = e[j];
          for (let k = j; k <= i - 1; k++) V[k][j] -= f * e[k] + g * d[k];
          d[j] = V[i - 1][j];
          V[i][j] = 0;
        }
      }
      d[i] = h;
    }
    for (let i = 0; i < n - 1; i++) {
      V[n - 1][i] = V[i][i];
      V[i][i] = 1;
      const h = d[i + 1];
      if (h !== 0) {
        for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
        for (let j = 0; j <= i; j++) {
          let g = 0;
          for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
          for (let k = 0; k <= i; k++) V[k][j] -= g * d[k];
        }
      }
      for (let k = 0; k <= i; k++) V[k][i + 1] = 0;
    }
    for (let j = 0; j < n; j++) {
      d[j] = V[n - 1][j];
      V[n - 1][j] = 0;
    }
    V[n - 1][n - 1] = 1;
    e[0] = 0;

    // QL iteration
    for (let i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0;
    let f = 0;
    let tst1 = 0;
    const eps = Math.pow(2, -52);
    for (let l = 0; l < n; l++) {
      tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
      let m = l;
      while (m < n) {
        if (Math.abs(e[m]) <= eps * tst1) break;
        m++;
      }
      if (m > l) {
        let iter = 0;
        do {
          iter++;
          if (iter > 200) throw new Error("tql2 did not converge");
          let g = d[l];
          let p = (d[l + 1] - g) / (2 * e[l]);
          let r = Math.hypot(p, 1);
          if (p < 0) r = -r;
          d[l] = e[l] / (p + r);
          d[l + 1] = e[l] * (p + r);
          const dl1 = d[l + 1];
          let h = g - d[l];
          for (let i = l + 2; i < n; i++) d[i] -= h;
          f += h;
          p = d[m];
          let c = 1;
          let c2 = c;
          let c3 = c;
          const el1 = e[l + 1];
          let s = 0;
          let s2 = 0;
          for (let i = m - 1; i >= l; i--) {
            c3 = c2;
            c2 = c;
            s2 = s;
            g = c * e[i];
            h = c * p;
            r = Math.hypot(p, e[i]);
            e[i + 1] = s * r;
            s = e[i] / r;
            c = p / r;
            p = c * d[i] - s * g;
            d[i + 1] = h + s * (c * g + s * d[i]);
            for (let k = 0; k < n; k++) {
              h = V[k][i + 1];
              V[k][i + 1] = s * V[k][i] + c * h;
              V[k][i] = c * V[k][i] - s * h;
            }
          }
          p = (-s * s2 * c3 * el1 * e[l]) / dl1;
          e[l] = s * p;
          d[l] = c * p;
        } while (Math.abs(e[l]) > eps * tst1);
      }
      d[l] += f;
      e[l] = 0;
    }
    const vectors = Array.from({ length: n }, (_, k) => {
      const col = new Float64Array(n);
      for (let i = 0; i < n; i++) col[i] = V[i][k];
      return col;
    });
    return { values: d, vectors };
  }

  // ---------------------------------------------------------------- eigensolver

  // Small deterministic PRNG so results are reproducible (mulberry32).
  function makeRng(seed) {
    let t = seed >>> 0;
    return function () {
      t = (t + 0x6d2b79f5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * Lowest eigenpairs of A x = lambda B x (A symmetric PSD, B diagonal positive) via a block
   * Krylov space of (A + shift B)^-1 B. Returns the `count` smallest Ritz pairs, ascending.
   */
  function lowestEigenpairs(op, count, options) {
    const opts = Object.assign({ block: 6, shift: 1, tol: 1e-7, maxDim: 700, seed: 12345 }, options || {});
    const csr = toCSR(op.rows);
    const n = csr.n;
    const bdiag = op.bdiag;
    const bw = bandwidth(csr);
    const shiftDiag = new Float64Array(n);
    for (let i = 0; i < n; i++) shiftDiag[i] = opts.shift * bdiag[i];
    const fac = bandCholesky(csr, shiftDiag, bw);
    const rng = makeRng(opts.seed);

    const basis = []; // B-orthonormal vectors q_j
    const bBasis = []; // B q_j
    const abasis = []; // A q_j
    const H = []; // H[j][i] = q_i . A q_j for i <= j
    const tmp = new Float64Array(n);

    function bnorm(x) {
      let s = 0;
      for (let i = 0; i < n; i++) s += bdiag[i] * x[i] * x[i];
      return Math.sqrt(s);
    }

    // B-orthonormalize w against the basis (Gram-Schmidt with DGKS re-orthogonalization) and add
    // it. Returns false if w was (numerically) in the span of the basis.
    function addToBasis(w) {
      const original = bnorm(w);
      let before = original;
      for (let pass = 0; pass < 3; pass++) {
        for (let j = 0; j < basis.length; j++) {
          const bq = bBasis[j];
          const q = basis[j];
          let c = 0;
          for (let i = 0; i < n; i++) c += bq[i] * w[i];
          for (let i = 0; i < n; i++) w[i] -= c * q[i];
        }
        const after = bnorm(w);
        if (after > 0.7 * before) break;
        before = after;
      }
      const norm = bnorm(w);
      if (!(norm > 1e-8 * original)) return false;
      for (let i = 0; i < n; i++) w[i] /= norm;
      const bq = new Float64Array(n);
      for (let i = 0; i < n; i++) bq[i] = bdiag[i] * w[i];
      const aw = new Float64Array(n);
      csrMul(csr, w, aw);
      const row = new Float64Array(basis.length + 1);
      for (let j = 0; j < basis.length; j++) {
        const q = basis[j];
        let s = 0;
        for (let i = 0; i < n; i++) s += q[i] * aw[i];
        row[j] = s;
      }
      let self = 0;
      for (let i = 0; i < n; i++) self += w[i] * aw[i];
      row[basis.length] = self;
      H.push(row);
      basis.push(w);
      bBasis.push(bq);
      abasis.push(aw);
      return true;
    }

    function extractRitz() {
      const d = basis.length;
      const M = Array.from({ length: d }, (_, i) => {
        const row = new Float64Array(d);
        for (let j = 0; j < d; j++) row[j] = j <= i ? H[i][j] : H[j][i];
        return row;
      });
      const { values, vectors } = symmetricEigen(M);
      const order = Array.from(values.keys()).sort((p, q) => values[p] - values[q]);
      const pairs = [];
      for (const k of order.slice(0, count)) {
        const x = new Float64Array(n);
        const Ax = new Float64Array(n);
        for (let j = 0; j < d; j++) {
          const c = vectors[k][j];
          const q = basis[j];
          const aq = abasis[j];
          for (let i = 0; i < n; i++) {
            x[i] += c * q[i];
            Ax[i] += c * aq[i];
          }
        }
        // residual in the B^-1 norm
        let r2 = 0;
        for (let i = 0; i < n; i++) {
          const r = Ax[i] - values[k] * bdiag[i] * x[i];
          r2 += (r * r) / bdiag[i];
        }
        pairs.push({ value: values[k], vector: x, residual: Math.sqrt(r2) });
      }
      return pairs;
    }

    // Start block, then expand block by block.
    let current = [];
    for (let b = 0; b < opts.block; b++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) w[i] = rng() - 0.5;
      if (addToBasis(w)) current.push(w);
    }
    let pairs = null;
    let sinceCheck = 0;
    while (basis.length < opts.maxDim) {
      const next = [];
      for (const v of current) {
        for (let i = 0; i < n; i++) tmp[i] = bdiag[i] * v[i];
        const w = new Float64Array(n);
        bandSolve(fac, tmp, w);
        if (addToBasis(w)) next.push(basis[basis.length - 1]);
      }
      current = next;
      if (current.length === 0) break;
      sinceCheck++;
      if (basis.length >= count + 3 && sinceCheck >= 3) {
        sinceCheck = 0;
        pairs = extractRitz();
        // converged when every wanted Ritz pair meets the tolerance (relative to its eigenvalue)
        let ok = pairs.length >= count;
        for (const p of pairs) if (!(p.residual < opts.tol * Math.max(1, Math.abs(p.value)))) ok = false;
        if (ok) break;
      }
    }
    if (pairs === null || basis.length >= opts.maxDim || current.length === 0) pairs = extractRitz();
    return { pairs, dimension: basis.length, bandwidth: bw };
  }

  /**
   * Chladni patterns: the `nPatterns` lowest plate modes with eigenvalue > 1 (rigid-body modes
   * dropped) on an n x n grid. Mirrors chladni_maths.calculate_patterns.
   * Returns { eigenvalues, frequencies, patterns } with patterns[i] a Float64Array (row-major
   * n x n, 2-norm 1, sign chosen so the entry of largest magnitude is positive).
   */
  function calculatePatterns(gridSize, mu, frequencyRatio, nPatterns, options) {
    const op = biharmonicOperator(gridSize, mu);
    const { pairs } = lowestEigenpairs(op, nPatterns + 5, options);
    const valid = pairs.filter((p) => p.value > 1).slice(0, nPatterns);
    if (valid.length < nPatterns) throw new Error("eigensolver did not return enough modes");
    const eigenvalues = Float64Array.from(valid.map((p) => p.value));
    const frequencies = eigenvalues.map((v) => frequencyRatio * Math.sqrt(v));
    const patterns = valid.map((p) => {
      const x = p.vector;
      let norm = 0;
      let big = 0;
      for (let i = 0; i < x.length; i++) {
        norm += x[i] * x[i];
        if (Math.abs(x[i]) > Math.abs(big)) big = x[i];
      }
      const s = (big < 0 ? -1 : 1) / Math.sqrt(norm);
      return x.map((v) => v * s);
    });
    return { eigenvalues, frequencies, patterns };
  }

  const api = {
    biharmonicOperator,
    lowestEigenpairs,
    calculatePatterns,
    toCSR,
    csrMul,
    bandwidth,
    bandCholesky,
    bandSolve,
    symmetricEigen,
    makeRng,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ChladniSolver = api;
})(typeof self !== "undefined" ? self : this);
