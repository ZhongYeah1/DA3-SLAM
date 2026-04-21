import os
import re
import numpy as np
import scipy
import time


def slice_with_overlap(lst, n, k):
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)  # Ensure progress even if k >= n
    return result


def sort_images_by_number(image_paths):
    def extract_number(path):
        filename = os.path.basename(path)
        # Look for digits followed immediately by a dot and the extension
        match = re.search(r'\d+(?:\.\d+)?(?=\.[^.]+$)', filename)
        return float(match.group()) if match else float('inf')

    return sorted(image_paths, key=extract_number)

def downsample_images(image_names, downsample_factor):
    """
    Downsamples a list of image names by keeping every `downsample_factor`-th image.

    Args:
        image_names (list of str): List of image filenames.
        downsample_factor (int): Factor to downsample the list. E.g., 2 keeps every other image.

    Returns:
        list of str: Downsampled list of image filenames.
    """
    return image_names[::downsample_factor]

def decompose_camera(P, no_inverse=False):
    """
    Decompose a 3x4 or 4x4 camera projection matrix P into intrinsics K,
    rotation R, and translation t.
    """
    if P.shape[0] != 3:
        P = P / P[-1,-1]
        P = P[0:3, :]

    # Ensure P is (3,4)
    assert P.shape == (3, 4)

    # Left 3x3 part
    M = P[:, :3]

    # RQ decomposition
    K, R = scipy.linalg.rq(M)

    # Make sure intrinsics have positive diagonal
    if K[0,0] < 0:
        K[:,0] *= -1
        R[0,:] *= -1
    if K[1,1] < 0:
        K[:,1] *= -1
        R[1,:] *= -1
    if K[2,2] < 0:
        K[:,2] *= -1
        R[2,:] *= -1

    scale = K[2,2]
    if not no_inverse:
        R = np.linalg.inv(R)
        t = -R @ np.linalg.inv(K) @ P[:, 3]
    else:
        t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale

    return K, R, t, scale

def normalize_to_sl4(H):
    """
    Normalize a 4x4 homography matrix H to be in SL(4).
    """
    det = np.linalg.det(H)
    if det == 0:
        raise ValueError("Homography matrix is singular and cannot be normalized.")
    scale = det ** (1/4)
    H_normalized = H / scale
    return H_normalized

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors a and b.
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T

class Accumulator:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.total_time += (time.perf_counter() - self.start)
