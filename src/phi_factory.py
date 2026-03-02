import gamma 
import numpy as np
from opt_einsum import contract

class PhiFactory:
    """
    Builds the elementals \phi for:
        - local operators
        - gamma_i vector operators
        - nabla derivative operators
        - chromomagnetic B / D operators
        - cubic orbit projected operators
    """

    def __init__(self, data):
        """
        data must provide:
            - get_elemental_block()
        """
        self.data = data
    
     # ==========================================================
    # Base gamma application (local / scalar / vector)
    # ==========================================================

    def _apply_gamma(self, op, D_block):
        """
        Apply base gamma structure.
        D_block shape: (4,4,nvec,nvec)
        """

        if op.gamma_i:
            # Sum_i gamma_i @ base_gamma
            return sum(
                contract("ij,ab->ijab",
                         gamma.gamma[i] @ op.base_gamma,
                         D_block)
                for i in range(1, 4)
            )
        else:
            return contract("ij,ab->ijab",
                            op.base_gamma,
                            D_block)

    # ==========================================================
    # Nabla derivative operator
    # ==========================================================

    def _phi_nabla(self, op, t, mom):

        gamma_i = [gamma.gamma[1],
                   gamma.gamma[2],
                   gamma.gamma[3]]

        D1 = self.data.get_elemental_block(mom, "disp_1")
        D2 = self.data.get_elemental_block(mom, "disp_2")
        D3 = self.data.get_elemental_block(mom, "disp_3")

        return sum(
            contract("ij,ab->ijab",
                     op.base_gamma @ gamma_i[i],
                     D1[t] if i == 0 else
                     D2[t] if i == 1 else
                     D3[t])
            for i in range(3)
        )

    # ==========================================================
    # Chromomagnetic B and D operators
    # ==========================================================

    def _phi_BD(self, op, t, mom):

        # B = antisymmetric combination
        # D = symmetric combination
        add = (op.derivative == "B")
        coeff = 1 if add else -1

        D = {
            "12": self.data.get_elemental_block(mom, "disp_1_2"),
            "21": self.data.get_elemental_block(mom, "disp_2_1"),
            "13": self.data.get_elemental_block(mom, "disp_1_3"),
            "31": self.data.get_elemental_block(mom, "disp_3_1"),
            "23": self.data.get_elemental_block(mom, "disp_2_3"),
            "32": self.data.get_elemental_block(mom, "disp_3_2"),
        }

        phi = (
            contract("ij,ab->ijab", gamma.gamma[1], D["23"][t]) -
            coeff * contract("ij,ab->ijab", gamma.gamma[1], D["32"][t]) +

            contract("ij,ab->ijab", gamma.gamma[2], D["31"][t]) -
            coeff * contract("ij,ab->ijab", gamma.gamma[2], D["13"][t]) +

            contract("ij,ab->ijab", gamma.gamma[3], D["12"][t]) -
            coeff * contract("ij,ab->ijab", gamma.gamma[3], D["21"][t])
        )

        return phi

    # ==========================================================
    # Single-momentum Φ builder
    # ==========================================================

    def _phi_single_momentum(self, op, t, mom):

        if op.derivative == "nabla":
            return self._phi_nabla(op, t, mom)

        elif op.derivative in {"B", "D"}:
            return self._phi_BD(op, t, mom)

        else:
            D = self.data.get_elemental_block(mom, "disp")
            return self._apply_gamma(op, D[t])

    # ==========================================================
    # Public Φ interface
    # ==========================================================

    def phi(self, op, t):
        """
        Unified phi builder eg. elemental with structure

        Handles:
            - local operators
            - derivatives
            - B/D operators
            - orbit projection
        """

        # ------------------------------------------------------
        # Single-momentum operator
        # ------------------------------------------------------
        if getattr(op, "orbit", None) is None:

            mom = op.mom
            if isinstance(mom, tuple):
                mom = f"mom_{mom[0]}_{mom[1]}_{mom[2]}"

            return self._phi_single_momentum(op, t, mom)

        # ------------------------------------------------------
        # Orbit-projected operator
        # ------------------------------------------------------
        total = None

        for p in op.orbit:

            mom = f"mom_{p[0]}_{p[1]}_{p[2]}"
            phi = self._phi_single_momentum(op, t, mom)

            if total is None:
                total = phi.copy()
            else:
                total += phi

        return total / len(op.orbit)