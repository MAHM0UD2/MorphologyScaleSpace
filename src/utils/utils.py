import numpy as np
import statistics
from image import Image
import pandas as pd
from src.state import state

def expand(u_t: Image) -> Image:
    target_h = 2 * u_t.height
    target_w = 2 * u_t.width

    # Initialize the target approximation image (Omega_{t-1})
    u_t_minus_one = Image(np.zeros((target_h, target_w), dtype=int))

    # Boundary Condition: Edge Replication (Zero-order hold)
    def get_p(r, c):
        r_clamp = max(0, min(r, u_t.height - 1))
        c_clamp = max(0, min(c, u_t.width - 1))
        return u_t[r_clamp, c_clamp]

    # Iterate over the TARGET grid (i, j)
    for i in range(target_h):
        for j in range(target_w):

            # Map to SOURCE grid (r, c)
            r = i // 2
            c = j // 2

            # 1. Base Pixel (Direct Copy)
            if i % 2 == 0 and j % 2 == 0:
                u_t_minus_one[i, j] = get_p(r, c)

            # 2. Horizontal Edge Interpolation
            elif i % 2 == 0 and j % 2 == 1:
                median_list = [
                    get_p(r-1, c),   get_p(r-1, c+1),
                    get_p(r, c),     get_p(r, c),     get_p(r, c),
                    get_p(r, c+1),   get_p(r, c+1),   get_p(r, c+1),
                    get_p(r+1, c),   get_p(r+1, c+1)
                ]
                u_t_minus_one[i, j] = statistics.median_low(median_list)

            # 3. Vertical Edge Interpolation
            elif i % 2 == 1 and j % 2 == 0:
                median_list = [
                    get_p(r, c-1),   get_p(r+1, c-1),
                    get_p(r, c),     get_p(r, c),     get_p(r, c),
                    get_p(r+1, c),   get_p(r+1, c),   get_p(r+1, c),
                    get_p(r, c+1),   get_p(r+1, c+1)
                ]
                u_t_minus_one[i, j] = statistics.median_low(median_list)

            # 4. Diagonal Center Interpolation
            else:
                median_list = [
                    get_p(r, c),     get_p(r+1, c),
                    get_p(r, c+1),   get_p(r+1, c+1)
                ]
                u_t_minus_one[i, j] = statistics.median_low(median_list)

    return u_t_minus_one

def decimate(u_t: Image) -> Image:
    # Base case: The image cannot be decimated further
    if u_t.width == 1 and u_t.height == 1:
        return u_t

    target_h = (u_t.height + 1) // 2
    target_w = (u_t.width + 1) // 2

    # Initialize the target decimated image (Omega_{t+1})
    u_t_plus_one = Image(np.zeros((target_h, target_w), dtype=int))

    # Iterate strictly over the smaller target grid (r, c)
    for r in range(target_h):
        for c in range(target_w):

            # Map back to the source grid (i, j)
            i = 2 * r
            j = 2 * c

            # Direct sub-sampling
            u_t_plus_one[r, c] = u_t[i, j]

    return u_t_plus_one

def compute_residual_t(u_t: Image, est_u_t: Image, t: int):
    # 1. Dimension Alignment: Crop est_u_t to match u_t strictly
    est_u_t_cropped = est_u_t.as_array()[:u_t.height, :u_t.width]

    # 2. Vectorized Subtraction
    r_t_array = u_t.as_array() - est_u_t_cropped

    # Temporary list to hold the new tuples for this level
    new_tuples = []

    # 3. Iteration and Non-Expansive Filtering
    for i in range(u_t.height):
        for j in range(u_t.width):
            # Skip the sub-sampled base pixels
            if i % 2 != 0 or j % 2 != 0:
                p = (i, j)
                v = r_t_array[i, j]

                new_tuples.append({"p": p, "v": v, "t": t})

    # 4. Batch Update the Feature Set
    if new_tuples:
        new_df = pd.DataFrame(new_tuples)
        # Using concat updates the global DataFrame reference
        state.F_1 = pd.concat([state.F_1, new_df], ignore_index=True)

def compute_a_t(t:int):
    a_t = state.F_1[state.F_1['t'] == t]['v'].unique()
    state.V_1[t] = a_t