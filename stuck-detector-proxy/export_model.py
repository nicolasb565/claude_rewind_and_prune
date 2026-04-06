#!/usr/bin/env python3
"""
One-time script: export sklearn model weights to JSON for the JS classifier.
Usage: python3 export_model.py [model.pkl] [output.json]
"""

import json
import pickle
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/nicolas/source/classifier-repos/dataset/stuck_classifier.pkl"
output_path = sys.argv[2] if len(sys.argv) > 2 else "model_weights.json"

with open(model_path, "rb") as f:
    data = pickle.load(f)

scaler = data["scaler"]
clf = data["classifier"]
feature_names = data["feature_names"]

weights = {
    "feature_names": feature_names,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coefficients": clf.coef_[0].tolist(),
    "intercept": float(clf.intercept_[0]),
}

with open(output_path, "w") as f:
    json.dump(weights, f, indent=2)

print(f"Exported {len(feature_names)} features to {output_path}")
print(f"  Features: {feature_names}")
print(f"  Intercept: {weights['intercept']:.4f}")
for name, coef in zip(feature_names, weights["coefficients"]):
    print(f"  {name}: {coef:+.4f}")
