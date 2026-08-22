import random
from src.state import state
import numpy as np
from src.utils.utils import uncommitted_uniform_quantization, committed_ward_clustering

def step_function(t: int, operation: str, t_arr: np.ndarray, v_arr: np.ndarray, active_mask: np.ndarray):
    match operation:
        case "Sparsification":
            drop_mask = (t_arr == t) & (v_arr == 0) & active_mask
            num_dropped = np.sum(drop_mask)

            if num_dropped > 0:
                active_mask[drop_mask] = False
                state.ell += num_dropped

        case "Quantization":
            active_v = v_arr[(t_arr == t) & active_mask & (v_arr != 0)]
            a_t = sorted(np.unique(active_v))
            if len(a_t) <= 1:
                return

            # Options:
            # uncommitted_uniform_quantization(a_t, t, t_arr, v_arr, active_mask)
            committed_ward_clustering(a_t, t, t_arr, v_arr, active_mask)

            state.ell += 1


def compression(t_arr: np.ndarray, v_arr: np.ndarray, active_mask: np.ndarray):
    if state.L == state.ell:
        return False

    random_t = random.randint(1, state.last - 1)
    random_op = random.choice(["Sparsification", "Quantization"])
    step_function(random_t, random_op, t_arr, v_arr, active_mask)

    return True