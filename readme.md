# Chladni Pattern Simulation
This project simulates Chladni figures, which are visual patterns formed by vibrating plates.

## Installation
```bash
pip install numpy scipy matplotlib pyqt6 pygame-ce
```

## Usage
- Run `show_patterns.py` to view patterns.
- Run `simulate_plate.py` to start an interactive simulation.

## Reference
Based on **Chladni Figures and the Tacoma Bridge: Motivating PDE Eigenvalue Problems via Vibrating Plates** by **Martin J. Gander and Felix Kwok**.

## Play in the browser
A JavaScript port (no installation, no dependencies) lives in [`docs/`](docs/): open `docs/index.html`, or
use the hosted version (GitHub Pages: Settings → Pages → Branch `main`, folder `/docs`).

- `docs/solver.js` – plate eigenmodes (port of `chladni_maths.py`; banded Cholesky + block Krylov eigensolver)
- `docs/simulation.js` – particle simulation (port of `chladni_simulation.py`)
- `docs/tests/` – `node --test docs/tests/` (Node ≥ 18). `reference.json` is generated from the Python code with
  `python docs/tests/make_reference.py`.
