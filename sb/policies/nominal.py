"""The nominal demonstrator: PD (kp = 6) on the geodesic through the marginal means,
one step of lookahead. Deterministic in the state, which is what makes clean demo labels
recoverable by re-evaluating it on recorded poses.
"""
from sb.core import se2 as S
from sb.envs import task as TK


class Nominal:
    def __init__(self, tk, kp=6.0):
        self.tk, self.kp = tk, kp

    def __call__(self, g, k, tau, step):
        a = self.tk.means[k].unsqueeze(0).expand_as(g)
        b = self.tk.means[k + 1].unsqueeze(0).expand_as(g)
        ref = S.SE2.interp(a, b, (tau + 1.0 / TK.T_SKILL).clamp(max=1.0))
        return self.kp * S.between(g, ref), None
