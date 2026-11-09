"""Remark 5.3, Appendix C.3 and Table 6: comparison of F7 with the 43 seven-party inequalities of the works listed in Table 6, by exact canonical forms under the 8! relabelings of A, ..., G and the purifier (after division by the gcd of the coefficients), and the weight bound for members of the toric, projective-plane and cyclic families with 8 to 14 parties.

The inequalities are transcribed from the cited works; the families are generated from their defining formulas (toric and projective-plane: Czech, Shuai, Wang, Zhang, eqs. (4) and (6), as also written in Hubeny and Yu; cyclic: Bao et al., eq. (4.10))."""
import sys
from math import gcd

import numpy as np

import inequalities as I
from core import canonical_form, merge_pair, parse

PRINTED = [
    ("Czech, Wang, eq. (2.1)", "ABDE + ABDF + ABEG + ADEF + ADEG + ACDE + ACDF + ACEG + BDEF + BDEG + BCDE + BCDF + BCEG + CDEF + CDEG >= ABC + ADE + ADF + AEG + BDE + BDF + BEG + CDE + CDF + CEG + ABDEF + ABDEG + ACDEF + ACDEG + BCDEF + BCDEG"),
    ("Czech, Wang, eq. (3.3)", "CFGO + CEGO + CDFO + BCGO + BCFO + BFGO + BEGO + BDFO + BDEF + BDEG + BCDE + BCDF + BCEG + CDEF + CDEG >= BDE + BDF + BEG + BFO + BGO + CDE + CDF + CEG + CFO + CGO + BCDEF + BCDEG + BCDFO + BCEGO + BCFGO + DEFGO"),
    ("Akers, Hernandez-Cuenca, Rath, eq. (1.5)", "ABDG + ACEG + BCFG >= A + B + C + ABCDEFG"),
    ("Bao, Naskar, eq. (A.11)", "ABCD + ABCG + ABFG + AEFG + BCDE + CDEF + DEFG >= ABC + ABG + AFG + BCD + CDE + DEF + EFG + ABCDEFG"),
    ("Bao, Naskar, eq. (A.12)", "ABDE + ABDF + ABEG + ACDE + ACDF + ACEG + ADEF + ADEG + BCDE + BCDF + BCEG + BDEF + BDEG + CDEF + CDEG >= ABC + ADE + ADF + AEG + BDE + BDF + BEG + CDE + CDF + CEG + ABDEF + ABDEG + ACDEF + ACDEG + BCDEF + BCDEG"),
    ("Bao, Furuya, Naskar, eq. (3.2)", "ABCF + ABCG + ABDF + ABEG + ACD + ADE + BCE + BDE >= AB + AC + AD + BC + BE + DE + ABCDF + ABCEG + ABDEFG"),
    ("Chou, Lao, Yang, eq. (5.15)", "ABDE + ABDF + ABEG + ADEF + ADEG + ACDE + ACDF + ACEG + BDEF + BDEG + BCDE + BCDF + BCEG + CDEF + CDEG >= ABC + ADE + ADF + AEG + BDE + BDF + BEG + CDE + CDF + CEG + ABDEF + ABDEG + ACDEF + ACDEG + BCDEF + BCDEG"),
]


def run(X, i, k):
    return tuple(X[(i + t) % len(X)] for t in range(k))


def toric(m, n):
    """sum_{i,j} [S(A_i^+ B_j^-) - S(A_i^- B_j^-)] - S(A_1 ... A_m) >= 0 with runs of (m+1)/2, (m-1)/2 and (n-1)/2 cyclically consecutive points; B_n is the purifier; N = m + n - 1."""
    A, B = list(range(m)), list(range(m, m + n))
    ap, am, bm = (m + 1) // 2, (m - 1) // 2, (n - 1) // 2
    L = [(1, run(A, i, ap) + run(B, j, bm)) for i in range(m) for j in range(n)]
    R = [(1, run(A, i, am) + run(B, j, bm)) for i in range(m) for j in range(n)] + [(1, tuple(A))]
    return m + n - 1, L, R


def projective(m):
    """Twice: (1/2) sum_{i,j} [S(A_i^(j) B_{i+j-1}^(m-j)) + S(A_i^(j) B_{i+j}^(m-j))] - sum_{i,j} S(A_i^(j-1) B_{i+j-1}^(m-j)) - S(A_1 ... A_m) >= 0; B_m is the purifier; N = 2m - 1."""
    A, B = list(range(m)), list(range(m, 2 * m))
    L = [(1, run(A, i, j) + run(B, i + j - 1, m - j)) for i in range(m) for j in range(1, m + 1)] + [(1, run(A, i, j) + run(B, i + j, m - j)) for i in range(m) for j in range(1, m + 1)]
    R = [(2, run(A, i, j - 1) + run(B, i + j - 1, m - j)) for i in range(m) for j in range(1, m + 1)] + [(2, tuple(A))]
    return 2 * m - 1, L, R


def cyclic(n, k, l, N):
    """sum_i S(A_i ... A_{i+l-1} | A_{i+l} ... A_{i+k+l-1}) >= S(A_1 ... A_n) for n regions; for n = N the purifier is extra, for n = N + 1 the regions are all N + 1 points."""
    P = list(range(n))
    L = [(1, run(P, i, k + l)) for i in range(n)]
    R = [(1, run(P, i + l, k)) for i in range(n) if k] + [(1, tuple(P))]
    return N, L, R


def to_dict(N, L, R):
    full = (1 << (N + 1)) - 1
    q = {}
    for sgn, side in ((1, L), (-1, R)):
        for c, t in side:
            m = sum(1 << p for p in t)
            if m >> N & 1:
                m ^= full
            if m:
                q[m] = q.get(m, 0) + sgn * c
    return {m: c for m, c in q.items() if c}


def weights(q):
    g = gcd(*q.values())
    return sum(c for c in q.values() if c > 0) // g, -sum(c for c in q.values() if c < 0) // g


def main():
    cF, oF = canonical_form(I.F7, 7)
    assert oF == 6720
    rows = [(name, parse(t, 7)) for name, t in PRINTED]
    rows += [(f"toric family, (m, n) = ({m}, {8 - m})", to_dict(*toric(m, 8 - m))) for m in (7, 5, 3, 1)]
    rows += [("projective-plane family, m = 4", to_dict(*projective(4)))]
    cyc = [to_dict(*cyclic(n, k, l, 7)) for n in (7, 8) for k in range(5) for l in range(1, 9) if 2 * k + l <= n and k + l < n]
    rows += [(f"cyclic family, member {i + 1}", q) for i, q in enumerate(c for c in cyc if c)]
    assert len(rows) == 43
    print("43 inequalities: source, orbit size, largest coefficient after division by the gcd, equal to F7 up to relabeling")
    for name, q in rows:
        c, o = canonical_form(q, 7)
        g = gcd(*q.values())
        same = bool(np.array_equal(c, cF))
        assert not same
        print(f"  {name}: {o}, {max(abs(x) // g for x in q.values())}, {same}")
    print(f"F7: orbit {oF}, largest coefficient {max(abs(x) for x in I.F7.values())}, no pair of points always together: {merge_pair(I.F7, 7) is None}")

    # merging or deleting parties keeps every coefficient on its side, so it cannot increase |alpha| or |beta| of the primitive form
    a7, b7 = weights(I.F7)
    worst = []
    for Np in range(8, 15):
        members = [toric(m, Np + 1 - m) for m in range(1, Np + 1, 2) if (Np + 1 - m) % 2]
        if (Np + 1) % 2 == 0:
            members.append(projective((Np + 1) // 2))
        members += [cyclic(n, k, l, Np) for n in (Np, Np + 1) for k in range(n) for l in range(1, n) if 2 * k + l <= n and k + l < n]
        for N_, L, R in members:
            q = to_dict(N_, L, R)
            if q:
                wa, wb = weights(q)
                assert wb <= 50 or wa < a7, (Np, wa, wb)
                worst.append(wb)
    print(f"members with 8 to 14 parties: |beta| <= 50 or |alpha| < {a7} for each (largest |beta| {max(worst)}); F7 has |alpha| = {a7}, |beta| = {b7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
