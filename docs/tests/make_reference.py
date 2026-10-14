# Generates reference.json for the JS tests from the original Python implementation
# (../../chladni_maths.py). Run: python tests/make_reference.py
import json, sys, pathlib
import numpy as np

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[2]))
import chladni_maths as cm

ref = {"operators": [], "modes": []}

# Full operator for small grids (triplets), to test the matrix assembly entry by entry.
for (n, mu) in [(6, 0.3), (9, 0.75)]:
    (A, B) = cm.biharmonic_operator(n, mu)
    coo = A.tocoo()
    ref["operators"].append({
        "n": n, "mu": mu,
        "rows": coo.row.tolist(), "cols": coo.col.tolist(), "vals": coo.data.tolist(),
        "bdiag": B.diagonal().tolist(),
    })

# Eigenvalues for several grids / mu, eigenvectors only for small grids.
for (n, mu, k, with_vectors) in [(12, 0.3, 20, True), (20, 0.01, 30, True), (30, 0.5, 30, True),
                                 (50, 0.2, 30, False), (50, 0.5, 30, False), (50, 0.99, 30, False)]:
    # The original uses ARPACK with a random start vector and its own sanity assert can fire
    # occasionally; retry until it passes.
    for attempt in range(10):
        try:
            (ev, ef) = cm.calculate_eigenvalues_and_eigenfunctions(n, mu, k)
            break
        except AssertionError:
            print("python reference assertion fired, retrying", (n, mu))
    else:
        raise RuntimeError("reference failed")
    entry = {"n": n, "mu": mu, "k": k, "eigenvalues": ev.tolist()}
    if with_vectors:
        entry["eigenvectors"] = ef.tolist()
    ref["modes"].append(entry)

(here / "reference.json").write_text(json.dumps(ref))
print("wrote", here / "reference.json")
