"""Ingeet: Real-Time Bengali Sign Language Recognition System."""

import os
import warnings
import numpy as np

# Ensure MediaPipe, scipy, and modern protobuf compatibility
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if not hasattr(np, "long"):
        np.long = int
    if not hasattr(np, "ulong"):
        np.ulong = int

