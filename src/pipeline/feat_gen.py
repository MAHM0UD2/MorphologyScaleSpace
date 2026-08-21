from src.utils.utils import decimate, compute_a_t, expand, compute_residual_t
import pandas as pd
from src.state import state
from src.utils.image import Image

def feature_generator(image: Image):

    u_t = image
    t = 1

    while True:
        u_t_plus_one = decimate(u_t)

        # Base Case condition utilizing dimension check
        if u_t_plus_one.width == u_t.width and u_t_plus_one.height == u_t.height:
            new_tuples = []
            for i in range(u_t.height):
                for j in range(u_t.width):
                    new_tuples.append({"p": (i, j), "v": u_t[i, j], "t": t})

            if new_tuples:
                state.F_1 = pd.concat([state.F_1, pd.DataFrame(new_tuples)], ignore_index=True)

            compute_a_t(t)
            state.last = t
            break

        est_u_t = expand(u_t_plus_one)

        compute_residual_t(u_t, est_u_t, t)
        compute_a_t(t)

        t += 1
        u_t = u_t_plus_one

    # L calculation evaluating the Restricted Zeros assumption
    sigma_A_t = 0
    for i in range(1, state.last + 1):
        sigma_A_t += len(state.V_1[i])

    zeros_count = len(state.F_1[state.F_1["v"] == 0])
    state.L = (zeros_count - 1) + (sigma_A_t - 1) - 1