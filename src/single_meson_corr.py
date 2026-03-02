import numpy as np
from opt_einsum import contract

class SingleMesonCorrelator:

    @staticmethod
    def two_pt_corr(proc, op_src, op_snk):
        perams = proc.perambulators()
        LT = proc.lt
        ntsrc = proc.ntsrc
        tsrc_step = proc.tsrc_step
        
        # Flavor selection
        flavor = op_src.meson

        flavor_map = {
            "light_light": ("light_fwd", "light_bwd"),
            "charm_charm": ("charm_fwd", "charm_bwd"),
            "charm_light": ("light_fwd", "charm_bwd"),
            "light_charm": ("light_fwd", "charm_bwd"),
        }

        try:
            fwd_key, bwd_key = flavor_map[flavor]
        except KeyError:
            raise NotImplementedError(f"Unsupported flavor: {flavor}")

        fwd = perams[fwd_key]
        bwd = perams[bwd_key]


        corr = np.zeros((ntsrc, LT), dtype=np.complex128)

        # Main contraction
        for isrc in range(ntsrc):
            tsrc = isrc * tsrc_step

            # Build source phi at physical source time
            phi_src = proc.phi(op_src, tsrc)
            for t in range(LT):
                tsnk = (tsrc + t) % LT
                # Sink phi at physical sink time
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