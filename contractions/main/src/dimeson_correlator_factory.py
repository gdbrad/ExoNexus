# dimeson_corr.py

import numpy as np
from opt_einsum import contract


class DiMesonCorrelator:

    @staticmethod
    def compute(proc, op1_src, op2_src, op1_snk, op2_snk):

        perams = proc.perambulators()
        LT, ntsrc = proc.lt, proc.ntsrc

        f1, f2 = proc.flavor_contents

        flavor_map = {
            "light_light": (perams["light_fwd"], perams["light_bwd"]),
            "light_charm": (perams["charm_fwd"], perams["light_bwd"]),
            "charm_light": (perams["light_fwd"], perams["charm_bwd"]),
            "charm_charm": (perams["charm_fwd"], perams["charm_bwd"]),
        }

        m1_fwd, m1_bwd = flavor_map[f1]
        m2_fwd, m2_bwd = flavor_map[f2]

        direct = np.zeros((ntsrc, LT))
        crossing = np.zeros((ntsrc, LT))

        phi0_1 = proc.phi(op1_src, 0)
        phi0_2 = proc.phi(op2_src, 0)

        for ts in range(ntsrc):
            for t in range(LT):

                phi_t_1 = proc.phi(op1_snk, t)
                phi_t_2 = proc.phi(op2_snk, t)

                c1 = contract("ijab,jkbc,klcd,lida",
                              phi_t_1,
                              m1_fwd[ts, t],
                              phi0_1,
                              m1_bwd[ts, t])

                c2 = contract("ijab,jkbc,klcd,lida",
                              phi_t_2,
                              m2_fwd[ts, t],
                              phi0_2,
                              m2_bwd[ts, t])

                direct[ts, t] = (c1 * c2).real

                m1x = contract("ijab,jkbc,klcd,lida",
                               phi_t_1,
                               m1_fwd[ts, t],
                               phi0_1,
                               m2_bwd[ts, t])

                m2x = contract("ijab,jkbc,klcd,lida",
                               phi_t_2,
                               m2_fwd[ts, t],
                               phi0_2,
                               m1_bwd[ts, t])

                crossing[ts, t] = (m1x * m2x).real

        return {
            "direct": direct,
            "crossing": crossing,
            "c15": direct - crossing,
            "c6": direct + crossing,
        }