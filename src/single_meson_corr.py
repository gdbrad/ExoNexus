# single_meson_corr.py

import numpy as np
from opt_einsum import contract


class SingleMesonCorrelator:

    @staticmethod
    def two_pt_corr(proc, op_src, op_snk):

        perams = proc.perambulators()

        LT = proc.lt
        ntsrc = proc.ntsrc
        tsrc_step = proc.tsrc_step

        # ------------------------------
        # Flavor selection
        # ------------------------------

        flavor = proc.flavor_contents[0]

        if flavor == "light_light":
            fwd = perams["light_fwd"]
            bwd = perams["light_bwd"]

        elif flavor == "charm_charm":
            fwd = perams["charm_fwd"]
            bwd = perams["charm_bwd"]

        elif flavor in ("charm_light", "light_charm"):
            fwd = perams["light_fwd"]
            bwd = perams["charm_bwd"]

        else:
            raise NotImplementedError(f"Unsupported flavor: {flavor}")

        corr = np.zeros((ntsrc, LT), dtype=np.complex128)

        # ------------------------------
        # Main contraction
        # ------------------------------

        for isrc in range(ntsrc):

            tsrc = isrc * tsrc_step

            # Build source Φ at physical source time
            phi_src = proc.phi(op_src, tsrc)

            for t in range(LT):

                tsnk = (tsrc + t) % LT

                # Sink Φ at physical sink time
                phi_snk = proc.phi(op_snk, tsnk)

                # Perambulators already indexed by (isrc, t)
                tau_f = fwd[isrc, t]
                tau_b = bwd[isrc, t]

                corr[isrc, t] = contract(
                    "ijab,jkbc,klcd,lida->",
                    phi_snk,
                    tau_f,
                    phi_src,
                    tau_b,
                    optimize="optimal"
                )

        return corr