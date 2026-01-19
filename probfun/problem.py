from pymoo.core.problem import Problem
import numpy as np



class DEB4(Problem):
    def __init__(self, n_var=30):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        ratio = np.clip(f1 / (g + 1e-8), 0.0, 1.0)
        f2 = g * (1.0 - (ratio + 1e-8) ** 0.25)
        out["F"] = np.column_stack([f1, f2])





class FON(Problem):
    def __init__(self, n_var=3):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=-4.0 * np.ones(n_var),
            xu=4.0 * np.ones(n_var)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        inv_sqrt_3 = 1.0 / np.sqrt(3.0)
        f1 = 1.0 - np.exp(-np.sum((x - inv_sqrt_3)**2, axis=1))
        f2 = 1.0 - np.exp(-np.sum((x + inv_sqrt_3)**2, axis=1))
        out["F"] = np.column_stack([f1, f2])







class SCH(Problem):
    def __init__(self):
        super().__init__(
            n_var=1,
            n_obj=2,
            n_constr=0,
            xl=-5.0,
            xu=5.0
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] ** 2
        f2 = (x[:, 0] - 2.0) ** 2
        out["F"] = np.column_stack([f1, f2])






class OKA2(Problem):
    def __init__(self, n_var=2):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(n_var),      # x_i ∈ [0,1]
            xu=np.ones(n_var)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x1 = x[:, 0]
        x2 = x[:, 1]

        f1 = x1
        f2 = 1 - x1**2 + 2 * np.sin(2 * np.pi * x1) * (x2 - 0.5)**2

        out["F"] = np.column_stack([f1, f2])