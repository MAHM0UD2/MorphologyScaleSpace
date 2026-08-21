from src.utils.utils import generate_r_t, expand, generate_u_last
from src.state import state
from src.utils.image import Image


def reconstruct() -> Image:
    u_current = generate_u_last()

    for t in range(state.last - 1, 0, -1):
      est_u = expand(u_current)
      r_t = generate_r_t(t, est_u.width, est_u.height)
      u_current = est_u + r_t

    return u_current