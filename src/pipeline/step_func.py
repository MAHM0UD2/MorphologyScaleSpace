import random
from src.state import state
from src.utils.utils import uncommitted_uniform_quantization, committed_ward_clustering

def step_function(t: int, operation: str):
    match operation:
        case "Sparsification":
            f_t_0 = state.F_1[(state.F_1["t"] == t) & (state.F_1["v"] == 0)]

            if not f_t_0.empty:
                index_to_drop = f_t_0.index[0]
                state.F_1.drop(index_to_drop, inplace=True)
                state.ell += 1

        case "Quantization":
            a_t = sorted(state.V_1[t])
            if len(a_t) <= 1:
                return

            # Options:
            uncommitted_uniform_quantization(a_t, t)
            # committed_ward_clustering(A_t, t)

            state.ell += 1


def compression():
    if state.L == state.ell:
        return False

    random_t = random.randint(1, state.last - 1)
    random_op = random.choice(["Sparsification", "Quantization"])
    step_function(random_t, random_op)

    return True