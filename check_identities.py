"""Exact identities between coefficient vectors: Proposition 3.2(i) and Remark 3.5 (B7* and B7 as nonnegative combinations of instances of subadditivity and MMI), Proposition 4.3(i) (6 U6 as a sum of the orbit sums of MMI), the decomposition of 42 R8 into 147 instances of MMI (Section 4.3), the descent (4.10) and Proposition 4.12, the identities (5.5) and (5.6) of Proposition 5.4, and the identities (5.10) of Remark 5.9 with their consequence for E0, ..., E5.

The instances are read from certificates/identities.json, rebuilt from the definitions of subadditivity and MMI (with purification), and summed in exact rational arithmetic."""
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import inequalities as I
from core import add, mmi, relabel, sa

CERT = Path(__file__).with_name("certificates") / "identities.json"


def combination(instances, N):
    """Sum of coefficient x instance, with a check that each instance has disjoint nonempty parts."""
    total = {}
    for kind, *parts, c in instances:
        letters = "".join(parts)
        assert len(set(letters)) == len(letters) and all(parts), parts
        assert (kind, len(parts)) in (("SA", 2), ("MMI", 3)), kind
        inst = sa(*parts, N) if kind == "SA" else mmi(*parts, N)
        total = add((1, total), (Fraction(c), inst))
    return total


def main():
    cert = json.loads(CERT.read_text())

    for name, n_inst, n_sa, n_mmi in (("B7*", 43, 26, 17), ("B7", 46, None, None)):
        inst = cert[name]
        coeffs = {Fraction(t[-1]) for t in inst}
        kinds = Counter(t[0] for t in inst)
        assert len(inst) == n_inst and all(c > 0 and (2 * c).denominator == 1 for c in coeffs)
        assert combination(inst, 7) == I.ALL[name][0], name
        if n_sa is not None:
            assert kinds == {"SA": n_sa, "MMI": n_mmi} and coeffs == {Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)}
        print(f"{name} = sum of {len(inst)} instances ({kinds['SA']} of subadditivity, {kinds['MMI']} of MMI) with coefficients {sorted(str(c) for c in coeffs)}: exact")

    psi = I.PSI
    assert add((6, I.U6)) == add((3, psi[0]), (3, psi[1]), (1, psi[2]), (1, psi[3]))
    print("Proposition 4.3(i): 6 U6 = 3 psi1 + 3 psi2 + psi3 + psi4, with", [len(x) for x in I.PSI_INSTANCES], "instances of MMI: exact")

    inst = cert["42 R8"]
    assert len(inst) == 147 and all("A" not in "".join(t[1:4]) for t in inst)
    assert combination(inst, 8) == add((42, I.R8))
    print("42 R8 = sum of 147 instances of MMI with coefficients (number of instances):", dict(sorted(Counter(int(t[-1]) for t in inst).items())), ": exact")

    assert add((3, I.Q8), (-1, I.Q8P), (-2, I.R8)) == {}
    print("Q8 = Q8'/3 + 2 R8/3, Q8'' = Q8' - 3 MMI(A:H:DE): exact (definitions)")

    E, MU = I.E, I.MU
    assert E[0] == add((1, E[5]), (2, MU[0]), (2, MU[1]), (1, MU[2]), (1, MU[3]), (2, MU[4]))
    assert all(m in E[1] and (E[1][m] > 0) == (c > 0) for m, c in MU[0].items())
    print("every term of mu0 is a term of E1 on the same side (Section 6.1)")
    print("Proposition 4.12: E0 = E5 + 2 mu0 + 2 mu1 + mu2 + mu3 + 2 mu4: exact")

    g = I.G
    gV7 = relabel(I.V7, g, 7)
    assert relabel(I.F7, g, 7) == I.F7 and gV7 != I.V7
    rhs55 = add((3, I.F7), (2, I.V7), (1, gV7), (2, I.V7P))
    assert add((4, I.Y7)) == rhs55
    print("(5.5): 4 Y7 = 3 F7 + 2 V7 + g(V7) + 2 V7' with g = (DF)(EG), Y7 = 6 E5 - 2 X7 - 3 mu5: exact")
    rhs56 = add((1, rhs55), (8, I.X7), (12, MU[5]), (48, MU[4]), (24, MU[3]), (24, MU[2]), (48, MU[1]), (48, MU[0]))
    assert add((24, E[0])) == rhs56
    print("(5.6): 24 E0 = 3 F7 + 2 V7 + g(V7) + 2 V7' + 8 X7 + 12 mu5 + 48 mu4 + 24 mu3 + 24 mu2 + 48 mu1 + 48 mu0: exact")

    gP7 = relabel(I.P7, g, 7)
    assert add((2, I.X7)) == add((1, I.V7P), (1, I.P7), (2, gP7))
    M5 = add((1, mmi("D", "BG", "CF", 7)), (1, mmi("E", "BF", "CG", 7)))
    assert add((2, E[5])) == add((1, I.V7P), (1, I.P7), (2, M5))
    print("(5.10): 2 X7 = V7' + P7 + 2 g(P7) and 2 E5 = V7' + P7 + 2 MMI(D:BG:CF) + 2 MMI(E:BF:CG): exact")
    steps = (2, 2, 1, 1, 2)
    for s in range(6):
        Ms = add((1, M5), *[(steps[t], MU[t]) for t in range(s, 5)])
        assert add((2, E[s])) == add((1, I.V7P), (1, I.P7), (2, Ms))
    print("Remark 5.9: 2 E_s = V7' + P7 + 2 M_s for s = 0, ..., 5, with M_s = MMI(D:BG:CF) + MMI(E:BF:CG) + sum over t >= s of (2, 2, 1, 1, 2)_t mu_t: exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
