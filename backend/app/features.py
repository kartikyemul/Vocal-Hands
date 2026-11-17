import numpy as np

def normalize(points):
    """21 (x,y,z) landmarks -> 63 features, translation + scale invariant."""
    p = np.asarray(points, dtype=float)
    p = p - p[0]                      # wrist at origin
    scale = np.linalg.norm(p, axis=1).max() or 1.0
    return (p / scale).flatten()
