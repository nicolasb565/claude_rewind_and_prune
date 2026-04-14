"""Generate MLP forward-pass parity test vectors for proxy/test/mlp.test.mjs.

Loads stuck_weights.json, generates 100 random inputs, runs the forward pass
in pure numpy (same math as the JS MLP), and writes the results to
proxy/test/mlp_parity_vectors.json.

Run from the repo root:
  python proxy/generate_test_vectors.py

The JS test (npm test) then verifies that mlp.mjs produces scores within 1e-5
of these reference values.
"""

import json
import os

import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "stuck_weights.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "test", "mlp_parity_vectors.json")

INPUT_DIM = 42  # no score history, no step_index_norm — every dim is a feature
N_VECTORS = 100
SEED = 42


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def forward(weights: dict, inp: np.ndarray) -> float:
    """Numpy forward pass — mirrors mlp.mjs exactly."""
    mean = np.array(weights["norm_mean"], dtype=np.float32)
    std = np.array(weights["norm_std"], dtype=np.float32)

    x = (inp - mean) / std

    h1 = np.maximum(0.0, np.array(weights["fc1.weight"], dtype=np.float32) @ x
                    + np.array(weights["fc1.bias"], dtype=np.float32))
    h2 = np.maximum(0.0, np.array(weights["fc2.weight"], dtype=np.float32) @ h1
                    + np.array(weights["fc2.bias"], dtype=np.float32))
    logit = (np.array(weights["fc3.weight"], dtype=np.float32) @ h2
             + np.array(weights["fc3.bias"], dtype=np.float32))[0]

    return float(sigmoid(logit))


def main() -> None:
    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        weights = json.load(f)

    rng = np.random.default_rng(SEED)
    vectors = []

    for _ in range(N_VECTORS):
        inp = rng.uniform(0.0, 1.0, INPUT_DIM).astype(np.float32)
        score = forward(weights, inp)
        vectors.append({"input": inp.tolist(), "score": score})

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vectors, f)

    print(f"Wrote {len(vectors)} vectors to {OUT_PATH}")


if __name__ == "__main__":
    main()
