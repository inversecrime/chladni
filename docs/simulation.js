// Chladni particle simulation: port of chladni_simulation.py (no pygame/numpy).
// Works in the browser (global `ChladniSimulation`) and in node (`require`).
// Units follow the original: positions in [0, 1]^2, time in milliseconds.

(function (root) {
  "use strict";

  const Solver = typeof require !== "undefined" && typeof module !== "undefined" ? require("./solver.js") : root.ChladniSolver;

  const AGE_MILESTONES = [2000, 7000, 12000];
  const COLOR_MILESTONES = [[0, 255, 0], [0, 0, 255], [255, 0, 0]];

  function clip(x, lo, hi) {
    return Math.min(Math.max(x, lo), hi);
  }

  class Simulation {
    constructor(options) {
      const o = Object.assign(
        {
          gridSize: 50,
          frequencyRatio: 8.0,
          nPatterns: 30,
          particleSize: 1,
          maxAge: Infinity,
          rng: Math.random,
        },
        options || {}
      );
      this.gridSize = o.gridSize;
      this.frequencyRatio = o.frequencyRatio;
      this.nPatterns = o.nPatterns;
      this.particleSize = o.particleSize;
      this.maxAge = o.maxAge;
      this.rng = o.rng;

      this.frequencies = null;
      this.patterns = null;
      this.mu = null;
      this.frequency = null;
      this.pattern = new Float64Array(this.gridSize * this.gridSize).fill(1);

      // structure of arrays; the first `count` entries are alive
      this.count = 0;
      this.capacity = 1024;
      this.x = new Float64Array(this.capacity);
      this.y = new Float64Array(this.capacity);
      this.speed = new Float64Array(this.capacity);
      this.direction = new Float64Array(this.capacity);
      this.age = new Float64Array(this.capacity);
    }

    get nParticles() {
      return this.count;
    }
    get minFrequency() {
      return this.frequencies[0];
    }
    get maxFrequency() {
      return this.frequencies[this.frequencies.length - 1];
    }

    /** |pattern| normalized by its mean, cubed, scaled: speed in units per millisecond. */
    scalePatternToSpeed(pattern) {
      const out = new Float64Array(pattern.length);
      let mean = 0;
      for (let i = 0; i < pattern.length; i++) mean += Math.abs(pattern[i]);
      mean /= pattern.length;
      for (let i = 0; i < pattern.length; i++) {
        const y = Math.abs(pattern[i]) / mean + 0.1;
        out[i] = (y * y * y * 0.3) / 1000;
      }
      return out;
    }

    /** Bilinear interpolation of the speed grid (first axis = x, second axis = y), like scipy's
     *  RegularGridInterpolator on linspace(0, 1, gridSize). */
    updateParticleSpeeds() {
      const n = this.gridSize;
      const speeds = this.scalePatternToSpeed(this.pattern);
      for (let p = 0; p < this.count; p++) {
        const fx = clip(this.x[p], 0, 1) * (n - 1);
        const fy = clip(this.y[p], 0, 1) * (n - 1);
        const i = Math.min(Math.floor(fx), n - 2);
        const j = Math.min(Math.floor(fy), n - 2);
        const tx = fx - i;
        const ty = fy - j;
        this.speed[p] =
          (1 - tx) * (1 - ty) * speeds[i * n + j] +
          (1 - tx) * ty * speeds[i * n + j + 1] +
          tx * (1 - ty) * speeds[(i + 1) * n + j] +
          tx * ty * speeds[(i + 1) * n + j + 1];
        this.direction[p] = this.rng() * 2 * Math.PI;
      }
    }

    /** Computes the plate modes for this mu (blocks for about a second at gridSize 50). */
    setMu(mu) {
      mu = clip(mu, 0.01, 0.99);
      const { patterns, frequencies } = Solver.calculatePatterns(this.gridSize, mu, this.frequencyRatio, this.nPatterns);
      this.setModes(patterns, frequencies);
      this.mu = mu;
      return mu;
    }

    /** Installs precomputed modes (ascending frequencies) and refreshes the current pattern. */
    setModes(patterns, frequencies) {
      this.patterns = patterns;
      this.frequencies = frequencies;
      if (this.frequency !== null) this.setFrequency(this.frequency);
    }

    /** Between two neighbouring resonances the plate is not in a mode: uniform pattern in the
     *  middle half, the nearer mode (blended) towards each resonance. */
    setFrequency(frequency) {
      if (this.frequencies === null) return frequency;
      frequency = clip(frequency, this.minFrequency, this.maxFrequency);

      let right = this.frequencies.findIndex((f) => f >= frequency);
      if (right < 0) right = this.frequencies.length - 1;
      const left = Math.max(right - 1, 0);

      if (left === right) {
        this.pattern = Float64Array.from(this.patterns[right]);
      } else {
        const fl = this.frequencies[left];
        const fr = this.frequencies[right];
        let t = (2 * (frequency - fl)) / (fr - fl) - 1;
        if (Math.abs(t) < 0.5) {
          this.pattern = new Float64Array(this.gridSize * this.gridSize).fill(1);
        } else {
          t = Math.sign(t) * (Math.abs(t) - 0.5) * 2;
          t = (t + 1) / 2;
          const pl = this.patterns[left];
          const pr = this.patterns[right];
          this.pattern = pl.map((v, i) => t * pr[i] + (1 - t) * v);
        }
      }
      this.frequency = frequency;
      return frequency;
    }

    grow(minCapacity) {
      if (minCapacity <= this.capacity) return;
      let cap = this.capacity;
      while (cap < minCapacity) cap *= 2;
      for (const key of ["x", "y", "speed", "direction", "age"]) {
        const bigger = new Float64Array(cap);
        bigger.set(this[key].subarray(0, this.count));
        this[key] = bigger;
      }
      this.capacity = cap;
    }

    throwParticles(n) {
      this.grow(this.count + n);
      for (let k = 0; k < n; k++) {
        const p = this.count + k;
        this.x[p] = this.rng();
        this.y[p] = this.rng();
        this.speed[p] = 0;
        this.direction[p] = 0;
        this.age[p] = 0;
      }
      this.count += n;
    }

    clearParticles() {
      this.count = 0;
    }

    /** Advance by dt milliseconds; particles that leave the plate (or get too old) disappear. */
    step(dt) {
      let kept = 0;
      for (let p = 0; p < this.count; p++) {
        const x = this.x[p] + dt * this.speed[p] * Math.cos(this.direction[p]);
        const y = this.y[p] + dt * this.speed[p] * Math.sin(this.direction[p]);
        const age = this.age[p] + dt;
        if (0 <= x && x <= 1 && 0 <= y && y <= 1 && age <= this.maxAge) {
          this.x[kept] = x;
          this.y[kept] = y;
          this.speed[kept] = this.speed[p];
          this.direction[kept] = this.direction[p];
          this.age[kept] = age;
          kept++;
        }
      }
      this.count = kept;
    }

    /** Color for a particle age (linear between the milestones, clamped). */
    static colorForAge(age) {
      const a = clip(age, AGE_MILESTONES[0], AGE_MILESTONES[AGE_MILESTONES.length - 1]);
      let k = 0;
      while (k < AGE_MILESTONES.length - 2 && a > AGE_MILESTONES[k + 1]) k++;
      const t = (a - AGE_MILESTONES[k]) / (AGE_MILESTONES[k + 1] - AGE_MILESTONES[k]);
      return COLOR_MILESTONES[k].map((c, i) => c + t * (COLOR_MILESTONES[k + 1][i] - c));
    }

    /** Draws into an RGBA buffer (e.g. ImageData.data) of size width x height: black background,
     *  one small disc per particle. */
    draw(rgba, width, height) {
      for (let i = 0; i < rgba.length; i += 4) {
        rgba[i] = 0;
        rgba[i + 1] = 0;
        rgba[i + 2] = 0;
        rgba[i + 3] = 255;
      }
      const r = this.particleSize;
      for (let p = 0; p < this.count; p++) {
        const [cr, cg, cb] = Simulation.colorForAge(this.age[p]);
        const px = Math.round(this.x[p] * width);
        const py = Math.round(this.y[p] * height);
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            if (dx * dx + dy * dy > r * r) continue;
            const sx = px + dx;
            const sy = py + dy;
            if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
            const o = (sy * width + sx) * 4;
            rgba[o] = cr;
            rgba[o + 1] = cg;
            rgba[o + 2] = cb;
          }
        }
      }
    }
  }

  const api = { Simulation, AGE_MILESTONES, COLOR_MILESTONES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ChladniSimulation = api;
})(typeof self !== "undefined" ? self : this);
