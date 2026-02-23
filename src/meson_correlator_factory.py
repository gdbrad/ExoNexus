# single_meson_corr.py

import numpy as np
from opt_einsum import contract

class SingleMesonCorrelator:

    @staticmethod
    def two_pt_corr(proc, op_src, op_snk):

        perams = proc.perambulators()
        LT, ntsrc = proc.lt, proc.ntsrc

        # Determine flavor
        flavor = proc.flavor_contents[0]

        if flavor == "light_light":
            fwd = perams["light_fwd"]
            bwd = perams["light_bwd"]

        elif flavor == "charm_charm":
            fwd = perams["charm_fwd"]
            bwd = perams["charm_bwd"]

        elif flavor == "charm_light":
            fwd = perams["light_fwd"]
            bwd = perams["charm_bwd"]

        else:
            raise NotImplementedError("Mixed flavor single meson?")

        corr = np.zeros((ntsrc, LT), dtype=np.complex128)

        phi0 = proc.phi(op_src, 0)

        for ts in range(ntsrc):
            for t in range(LT):

                phi_t = proc.phi(op_snk, t)

                corr[ts, t] = contract(
                    "ijab,jkbc,klcd,lida",
                    phi_t,
                    fwd[ts, t],
                    phi0,
                    bwd[ts, t]
                )

        return corr.real