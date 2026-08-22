import numpy as np
import statistics
from src.utils.image import Image
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
    # 1. Image Subtraction
    r_t_array = u_t - est_u_t

    # Temporary list to hold the new tuples for this level
    new_tuples = []

    # 2. Iteration and Non-Expansive Filtering
    for i in range(u_t.height):
        for j in range(u_t.width):
            # Skip the sub-sampled base pixels
            if i % 2 != 0 or j % 2 != 0:
                p = (i, j)
                v = r_t_array[i, j]

                new_tuples.append({"p": p, "v": v, "t": t})

    # 3. Batch Update the Feature Set
    if new_tuples:
        new_df = pd.DataFrame(new_tuples)
        # Using concat updates the global DataFrame reference
        state.F_1 = pd.concat([state.F_1, new_df], ignore_index=True)

def generate_r_t(t: int, width_t: int, height_t: int) -> Image:
    r_t_array = np.zeros((height_t, width_t), dtype=int)
    f_t = state.F_1.loc[state.F_1['t'] == t]

    for p, v in zip(f_t["p"], f_t["v"]):
        i, j = p
        r_t_array[i, j] = v

    return Image(r_t_array)

def compute_a_t(t: int):
    a_t = state.F_1[state.F_1['t'] == t]['v'].unique()
    state.V_1[t] = a_t

def uncommitted_uniform_quantization(a_t: list[int], t: int, t_arr: np.ndarray, v_arr: np.ndarray, active_mask: np.ndarray):
    min_diff = float('inf')
    merge_idx = 0

    for i in range(len(a_t) - 1):
        diff = a_t[i + 1] - a_t[i]
        if diff < min_diff:
            min_diff = diff
            merge_idx = i

    q1 = a_t[merge_idx]
    q2 = a_t[merge_idx + 1]
    s = q1 + q2
    # sgn(s) * ((|s| + 1) / 2): Go to nearest integer
    q_new = int(np.sign(s) * ((abs(s) + 1) // 2))

    mask_1 = (t_arr == t) & (v_arr == q1) & active_mask
    mask_2 = (t_arr == t) & (v_arr == q2) & active_mask

    v_arr[mask_1 | mask_2] = q_new



def committed_ward_clustering(a_t: list[int], t: int, t_arr: np.ndarray, v_arr: np.ndarray, active_mask: np.ndarray):
    best_q1, best_q2, best_q_new = None, None, None
    min_mse_increase = float('inf')

    for i in range(len(a_t) - 1):
        q1 = a_t[i]
        q2 = a_t[i + 1]

        mask_1 = (t_arr == t) & (v_arr == q1) & active_mask
        mask_2 = (t_arr == t) & (v_arr == q2) & active_mask

        count_q1 = np.sum(mask_1)
        count_q2 = np.sum(mask_2)
        q_new = q1 if count_q1 >= count_q2 else q2

        mse_increase = count_q1 * ((q1 - q_new) ** 2) + count_q2 * ((q2 - q_new) ** 2)

        if mse_increase < min_mse_increase:
            min_mse_increase = mse_increase
            best_q1 = q1
            best_q2 = q2
            best_q_new = q_new

    mask1 = (t_arr == t) & (v_arr == best_q1) & active_mask
    mask2 = (t_arr == t) & (v_arr == best_q2) & active_mask

    v_arr[mask1 | mask2] = best_q_new


def generate_u_last() -> Image:
    f_last = state.F_1.loc[state.F_1['t'] == state.last, ['p', 'v']]

    max_i = max(p[0] for p in f_last['p'])
    max_j = max(p[1] for p in f_last['p'])

    u_last_array = np.zeros((max_i + 1, max_j + 1), dtype=int)

    for p, v in zip(f_last["p"], f_last["v"]):
        i, j = p
        u_last_array[i, j] = v

    return Image(u_last_array)