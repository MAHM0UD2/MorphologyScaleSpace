from dataclasses import dataclass, field
import pandas as pd

@dataclass
class GlobalState:
    F_1: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["p","v","t"]))
    V_1: dict = field(default_factory=dict)
    L: int = 0
    last: int = 1
    ell: int = 0
    input_folder: str = r"G:\MorphologyScaleSpace\images"

    def reset(self):
        self.F_1 = field(default_factory=lambda: pd.DataFrame(columns=["p","v","t"]))
        self.V_1 = field(default_factory=dict)
        self.L = 0
        self.last = 1
        self.ell = 0

state = GlobalState()