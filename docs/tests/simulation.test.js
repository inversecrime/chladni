// Tests for simulation.js (particle simulation, frequency interpolation, drawing).
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { Simulation } = require("../simulation.js");
const { makeRng } = require("../solver.js");

const near = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

function fakeSim(n = 4) {
  const sim = new Simulation({ gridSize: n, nPatterns: 3, rng: makeRng(1) });
  const size = n * n;
  const patterns = [1, 2, 3].map((k) => Float64Array.from({ length: size }, (_, i) => k * (1 + (i % 3))));
  sim.setModes(patterns, Float64Array.from([10, 20, 40]));
  return sim;
}

test("initial pattern is uniform", () => {
  const sim = new Simulation({ gridSize: 5 });
  assert.equal(sim.pattern.length, 25);
  assert.ok(sim.pattern.every((v) => v === 1));
});

test("scalePatternToSpeed: |p| / mean(|p|), (y + 0.1)^3 * 0.3 / 1000", () => {
  const sim = new Simulation({ gridSize: 2 });
  const s = sim.scalePatternToSpeed(Float64Array.from([-1, 1, 3, 3])); // mean |p| = 2
  [0.5, 0.5, 1.5, 1.5].forEach((y, i) => near(s[i], (Math.pow(y + 0.1, 3) * 0.3) / 1000));
  const uniform = sim.scalePatternToSpeed(new Float64Array(4).fill(7));
  near(uniform[0], (Math.pow(1.1, 3) * 0.3) / 1000);
});

test("setFrequency before any modes exist is a no-op", () => {
  const sim = new Simulation({ gridSize: 3 });
  assert.equal(sim.setFrequency(5), 5);
  assert.equal(sim.frequency, null);
});

test("setFrequency clamps to the frequency range", () => {
  const sim = fakeSim();
  assert.equal(sim.setFrequency(-5), 10);
  assert.equal(sim.setFrequency(1e9), 40);
});

test("setFrequency: pure mode at a resonance, uniform pattern between resonances, blend near them", () => {
  const sim = fakeSim();
  const [p1, p2] = [sim.patterns[0], sim.patterns[1]];
  // exactly at the lower end: pattern of that mode
  sim.setFrequency(10);
  assert.deepEqual(Array.from(sim.pattern), Array.from(p1));
  // exactly at a resonance: that mode
  sim.setFrequency(20);
  assert.deepEqual(Array.from(sim.pattern), Array.from(p2));
  // middle half between 10 and 20 (12.5 .. 17.5): uniform
  for (const f of [13, 15, 17]) {
    sim.setFrequency(f);
    assert.ok(sim.pattern.every((v) => v === 1), `f=${f}`);
  }
  // 11: t = 2*(1/10) - 1 = -0.8 -> t = -(0.3*2) = -0.6 -> (t+1)/2 = 0.2 -> 0.2 p2 + 0.8 p1
  sim.setFrequency(11);
  sim.pattern.forEach((v, i) => near(v, 0.2 * p2[i] + 0.8 * p1[i]));
  // 19: symmetric near p2
  sim.setFrequency(19);
  sim.pattern.forEach((v, i) => near(v, 0.8 * p2[i] + 0.2 * p1[i]));
});

test("changing the modes refreshes the pattern for the current frequency", () => {
  const sim = fakeSim();
  sim.setFrequency(20);
  const other = [1, 2, 3].map((k) => new Float64Array(16).fill(-k));
  sim.setModes(other, Float64Array.from([10, 20, 40]));
  assert.ok(sim.pattern.every((v) => v === -2));
});

test("throwParticles adds uniformly distributed particles at rest and grows storage", () => {
  const sim = new Simulation({ gridSize: 3, rng: makeRng(7) });
  sim.throwParticles(5000);
  assert.equal(sim.nParticles, 5000);
  assert.ok(sim.capacity >= 5000);
  let sx = 0;
  for (let p = 0; p < 5000; p++) {
    assert.ok(sim.x[p] >= 0 && sim.x[p] < 1 && sim.y[p] >= 0 && sim.y[p] < 1);
    assert.equal(sim.speed[p], 0);
    assert.equal(sim.age[p], 0);
    sx += sim.x[p];
  }
  near(sx / 5000, 0.5, 0.02);
  sim.throwParticles(10);
  assert.equal(sim.nParticles, 5010);
});

test("step moves particles with speed and direction, ages them, and drops those leaving [0,1]^2", () => {
  const sim = new Simulation({ gridSize: 3 });
  sim.throwParticles(3);
  sim.x.set([0.5, 0.99, 0.5]);
  sim.y.set([0.5, 0.5, 0.01]);
  sim.speed.set([0.001, 0.001, 0.001]);
  sim.direction.set([0, 0, -Math.PI / 2]);
  sim.step(100); // moves by 0.1
  assert.equal(sim.nParticles, 1);
  near(sim.x[0], 0.6);
  near(sim.y[0], 0.5);
  near(sim.age[0], 100);
});

test("maxAge removes old particles", () => {
  const sim = new Simulation({ gridSize: 3, maxAge: 150 });
  sim.throwParticles(2);
  sim.step(100);
  assert.equal(sim.nParticles, 2);
  sim.step(100);
  assert.equal(sim.nParticles, 0);
});

test("updateParticleSpeeds interpolates bilinearly (x = first grid axis) and randomizes directions", () => {
  const n = 5;
  const sim = new Simulation({ gridSize: n, rng: makeRng(3) });
  // |pattern| = 1 + i (row index = x axis) -> speeds follow scalePatternToSpeed of that field
  const field = Float64Array.from({ length: n * n }, (_, k) => 1 + Math.floor(k / n));
  sim.pattern = field;
  const speeds = sim.scalePatternToSpeed(field);
  sim.throwParticles(3);
  sim.x.set([0, 0.25, 1]);
  sim.y.set([0.3, 0.9, 1]);
  sim.updateParticleSpeeds();
  near(sim.speed[0], speeds[0]); // grid row 0
  near(sim.speed[1], speeds[1 * n]); // x = 0.25 is exactly row 1 (any y: field is constant in y)
  near(sim.speed[2], speeds[4 * n]);
  // half way between rows 1 and 2
  sim.x[0] = 0.375;
  sim.updateParticleSpeeds();
  near(sim.speed[0], 0.5 * (speeds[1 * n] + speeds[2 * n]));
  for (let p = 0; p < 3; p++) assert.ok(sim.direction[p] >= 0 && sim.direction[p] < 2 * Math.PI);
});

test("colorForAge: green -> blue -> red, clamped", () => {
  assert.deepEqual(Simulation.colorForAge(0), [0, 255, 0]);
  assert.deepEqual(Simulation.colorForAge(2000), [0, 255, 0]);
  assert.deepEqual(Simulation.colorForAge(7000), [0, 0, 255]);
  assert.deepEqual(Simulation.colorForAge(12000), [255, 0, 0]);
  assert.deepEqual(Simulation.colorForAge(99999), [255, 0, 0]);
  const mid = Simulation.colorForAge(4500);
  near(mid[1], 127.5);
  near(mid[2], 127.5);
  const late = Simulation.colorForAge(9500);
  near(late[0], 127.5);
  near(late[2], 127.5);
});

test("draw: black background, particle disc in the age color, off-canvas pixels ignored", () => {
  const sim = new Simulation({ gridSize: 3, particleSize: 1 });
  sim.throwParticles(2);
  sim.x.set([0.5, 0.95]);
  sim.y.set([0.5, 0.95]);
  sim.age.set([0, 12000]);
  const w = 20;
  const h = 20;
  const buf = new Uint8ClampedArray(w * h * 4).fill(99);
  sim.draw(buf, w, h);
  const px = (x, y) => Array.from(buf.slice((y * w + x) * 4, (y * w + x) * 4 + 4));
  assert.deepEqual(px(0, 0), [0, 0, 0, 255]);
  assert.deepEqual(px(10, 10), [0, 255, 0, 255]); // young: green
  assert.deepEqual(px(11, 10), [0, 255, 0, 255]); // radius 1 plus shape
  assert.deepEqual(px(11, 11), [0, 0, 0, 255]); // diagonal not covered
  assert.deepEqual(px(19, 19), [255, 0, 0, 255]);
  assert.deepEqual(px(19, 18), [255, 0, 0, 255]); // old particle near the corner: red, clipped at the border
});

test("same seed gives the same simulation", () => {
  const run = () => {
    const sim = fakeSim();
    sim.rng = makeRng(42);
    sim.setFrequency(20);
    sim.throwParticles(100);
    for (let k = 0; k < 20; k++) {
      sim.updateParticleSpeeds();
      sim.step(16);
    }
    return Array.from(sim.x.subarray(0, sim.nParticles));
  };
  assert.deepEqual(run(), run());
});

test("setMu with the real solver gives sorted frequencies and a usable pattern", () => {
  const sim = new Simulation({ gridSize: 20, nPatterns: 10, rng: makeRng(2) });
  assert.equal(sim.setMu(5), 0.99); // clipped
  assert.equal(sim.setMu(-1), 0.01);
  assert.equal(sim.setMu(0.5), 0.5);
  for (let i = 1; i < 10; i++) assert.ok(sim.frequencies[i] >= sim.frequencies[i - 1]);
  sim.setFrequency(sim.frequencies[3]);
  assert.equal(sim.pattern.length, 400);
  sim.setMu(0.4); // frequency kept, pattern recomputed
  assert.equal(sim.pattern.length, 400);
  sim.throwParticles(50);
  sim.updateParticleSpeeds();
  assert.ok(Array.from(sim.speed.subarray(0, 50)).every((v) => v > 0));
});
