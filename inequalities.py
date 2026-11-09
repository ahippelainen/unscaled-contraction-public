"""The inequalities of the paper, transcribed from its displays or defined by the identities of its text, and the exact integer computations of their invariants (Sections 3-6, Tables 1-3, Appendix C).

Run as a script, it asserts every invariant the paper prints for these inequalities and checks that certificates/inequalities.txt, the full coefficient lists, is exactly what the definitions below produce."""
import itertools
import sys
from fractions import Fraction
from math import factorial, gcd
from pathlib import Path

import numpy as np

from core import (add, antichain_at_A, bell_values, delete_party, invariants, merge_pair, mmi, names, occurrence_words, parse, permutation, realizable_domain, relabel, sides, stabilizer, subsets, term_mask, text, u_domain)

# Section 3, eq. (3.1), and Remark 3.5
B7STAR = parse("2A + AO + 3(BO + CO + DO + EO + FO + GO) >= ABC + ABD + ACD + AEF + AEG + AFG + 14O", 7)
B7 = parse("3(AO + BO + CO + DO + EO + FO + GO) + DEFG >= ABC + ADE + AFG + BDEF + BDEG + CDFG + CEFG + 15O", 7)

# Section 4.1, eqs. (4.1) and (4.2)
U6 = parse("AB + AC + AD + AE + AF + BC + BD + BE + BF + CD + CE + CF + DE + DF + EF + ABC + DEF + ABCDE + ABCDF + ABCEF + ABDEF + ACDEF + BCDEF"
           " >= 4(A + B + C + D + E + F) + ABCD + ABCE + ABCF + ADEF + BDEF + CDEF + 3ABCDEF", 6)
TRIANGLES = ("ABC", "DEF")
EDGES = [a + b for t in TRIANGLES for a, b in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))]


def _other(t, x):
    return "".join(c for c in t if c not in x)


def _psi():
    """The orbit sums psi_1, ..., psi_4 of (4.2), as lists of the (X, Y, Z) of their MMI instances."""
    tri = {v: t for t in TRIANGLES for v in t}
    opp = {v: TRIANGLES[1 - TRIANGLES.index(tri[v])] for v in "ABCDEF"}
    psi1 = [(e[0], e[1], _other("ABCDEF", e)) for e in EDGES]
    psi2 = [tuple(TRIANGLES[0]), tuple(TRIANGLES[1])]
    psi3 = [(e[0], e[1], x) for e in EDGES for x in opp[e[0]]]
    psi4 = [(i, x, _other(tri[i], i)) for i in "ABCDEF" for x in opp[i]]
    return [psi1, psi2, psi3, psi4]


PSI_INSTANCES = _psi()
PSI = [add(*[(1, mmi(*t, 6)) for t in inst]) for inst in PSI_INSTANCES]

# Section 4.2, eq. (4.6); Section 4.3, eqs. (4.7), (4.8)
Q8A = parse("ADE + AFG + ABCH + ABDFH + ABEGH + ACDGH + ACEFH >= 2A + ABDGH + ABEFH + ACDFH + ACEGH + ABCDEFGH", 8)
Q8 = parse("ADE + AFG + ABCH + ABDFH + ABEGH + ACDGH + ACEFH + 2(BCDE + BCFG + BDGH + BEFH + CDFH + CEGH + DEFG)"
           " >= 2A + 4(B + C + D + E + F + G + H) + DE + FG + BCH + ABDGH + ABEFH + ACDFH + ACEGH + 3BCDEFGH + ABCDEFGH", 8)
R8 = parse("BCDE + BCFG + BDGH + BEFH + CDFH + CEGH + DEFG >= 2(B + C + D + E + F + G + H) + 2BCDEFGH", 8)
Q8P = add((3, Q8), (-2, R8))
Q8PP = add((1, Q8P), (-3, mmi("A", "H", "DE", 8)))
E0 = parse("3(ABC + ADE + AFG) + 4(BDG + BEF + CDF + CEG) + 3(ABDF + ABEG + ACDG + ACEF) + 4(BCDE + BCFG + DEFG)"
           " >= 6A + 8(B + C + D + E + F + G) + 3(BC + DE + FG) + 3(ABDG + ABEF + ACDF + ACEG) + 5BCDEFG + 3ABCDEFG", 7)
MU = [mmi("BC", "DE", "FG", 7),
      mmi("B", "D", "F", 7),
      add((1, mmi("A", "C", "DF", 7)), (1, mmi("A", "E", "BF", 7)), (1, mmi("A", "G", "BD", 7))),
      add((1, mmi("A", "B", "D", 7)), (1, mmi("A", "B", "F", 7)), (1, mmi("A", "D", "F", 7))),
      add((1, mmi("B", "E", "G", 7)), (1, mmi("C", "D", "G", 7)), (1, mmi("C", "E", "F", 7))),
      add((1, mmi("C", "DF", "EG", 7)), (1, mmi("E", "BF", "CG", 7)), (1, mmi("G", "BD", "CE", 7)))]
# eqs. (4.9), (4.10)
E1 = add((1, E0), (-2, MU[0]))
E2 = add((1, E1), (-2, MU[1]))
E3 = add((1, E2), (-1, MU[2]))
E4 = add((1, E3), (-1, MU[3]))
E5 = add((1, E4), (-2, MU[4]))
E = [E0, E1, E2, E3, E4, E5]

# Section 5, eq. (5.1), and Appendix C, eqs. (C.1)-(C.5)
F7 = parse("2ABC + 2ADE + 2AFG + BDF + 4BDG + 4BEF + BEG + 4CDF + CDG + CEF + 2ABDF + 2ABEG + 2ACDG + 2ACEF + BCDE + BCFG + DEFG + 2BCDEG + 2BCEFG + 2CDEFG"
           " >= 3B + 2C + 3D + 2E + 3F + 2G + 2AB + 2AD + 2AF + 3BE + 3BG + 3CD + 3CF + 3DG + 3EF + ABDG + ABEF + ACDF + 3ACEG + BCDEF + BCDFG + BDEFG + 4BCDEFG + 2ABCDEFG", 7)
X7 = parse("7ABC + 7ADE + 7AFG + 5BDF + 5BDG + 5BEF + 5BEG + 5CDF + 5CDG + 5CEF + 6CEG + 7ABDF + 7ABEG + 7ACDG + 7ACEF + 4BCDE + 4BCFG + 4DEFG"
           " >= 4B + 5C + 4D + 5E + 4F + 5G + 4AB + 3AC + 4AD + 3AE + 4AF + 3AG + 3BC + 4BD + 4BE + 4BF + 4BG + 4CD + 4CE + 4CF + 4CG + 3DE + 4DF + 4DG + 4EF + 4EG + 3FG"
           " + 5ABDG + 5ABEF + 5ACDF + 6ACEG + 5BCDEFG + 7ABCDEFG", 7)
V7 = parse("2ABC + 2ADE + 2AFG + BDF + 2BDG + 2BEF + BEG + 2CDF + CDG + CEF + 2CEG + 2ABDF + 2ABEG + 2ACDG + 2ACEF + 3BCFG + BCDEF + 2BCDEG + BDEFG + 2CDEFG"
           " >= B + 2C + 3D + 4E + F + 2G + 2AB + 2AD + 2AF + BE + 2BF + 3BG + CD + 3CF + 2CG + DG + EF + ABDG + ABEF + ACDF + 3ACEG + BCDE + DEFG + BCDFG + 4BCDEFG + 2ABCDEFG", 7)
V7P = parse("2ABC + 2ADE + 2AFG + BDF + BDG + BEF + BEG + CDF + CDG + CEF + 3CEG + 2ABDF + 2ABEG + 2ACDG + 2ACEF + 2DEFG + BCDEF + BCDEG + 2BCDFG + 2BCEFG"
            " >= 2B + 4C + 2E + F + 3G + 2AB + 2AD + 2AF + BD + BE + CD + CE + 2DF + 2DG + 2EF + 2EG + ABDG + ABEF + ACDF + 3ACEG + 2BCFG + 4BCDEFG + 2ABCDEFG", 7)
P7 = parse("4ABC + 4ADE + 4AFG + 3BDF + 3BDG + 3BEF + 3BEG + 3CDF + 3CDG + 3CEF + 3CEG + 4ABDF + 4ABEG + 4ACDG + 4ACEF + 4BCDE + 2BCFG + 2DEFG"
           " >= 2B + 2C + 2D + 2E + 3F + 3G + 2AB + 2AC + 2AD + 2AE + 2AF + 2AG + 2BC + 3BD + 3BE + 2BF + 2BG + 3CD + 3CE + 2CF + 2CG + 2DE + 2DF + 2DG + 2EF + 2EG + 2FG"
           " + 3ABDG + 3ABEF + 3ACDF + 3ACEG + BCDEF + BCDEG + 2BCDEFG + 4ABCDEFG", 7)
Y7 = add((6, E5), (-2, X7), (-3, MU[5]))
G = permutation("DF,EG", 7)
# Remark 5.9: the half-sum (V7' + P7)/2
_twice = add((1, V7P), (1, P7))
assert all(c % 2 == 0 for c in _twice.values()), "V7' + P7 is not even"
HALF = {m: c // 2 for m, c in _twice.items()}

# Section 6, eqs. (6.12), (6.18), (6.20), and Appendix C.2, eq. (C.6)
QPRIME = parse("A + B + C + ABC >= AB + AC", 3)
D0 = parse("2A + 2B + 2C + 2D + 2E + 2F + 3G + ABCDEFG >= ACEG + ADFG + BCFG + BDEG", 7)
Z7 = parse("4A + 3B + 4C + 4D + 4E + 3F + 4G + 2ABCDEFG >= ABCG + ABDE + ACEF + BCEF + BEFG + CDEG + CDFG", 7)
G9 = parse("3A + 5B + 5C + 5D + 3E + 4F + 6G + 6H + 4I + 4ABCDEFGHI >= ABCDG + ABCHI + ABEFG + ACDEF + ACEGH + BCDEH + BCFGH + CDEGI + DFGHI", 9)

# Appendix C.2: the lists of weighings of H9, H11, H15 and H17 (coins numbered from 1); for H15 and H17 the list is (T_1, ..., T_n, complement of T_1, ..., complement of T_n)
WEIGHINGS = {
    "H9": (8, [{5, 7, 8}, {1, 2, 3, 5, 7}, {2, 4, 5, 8}, {1, 4, 6, 7}, {1, 3, 4, 8}, {2, 3, 6, 7}, {1, 2, 6, 8}, {3, 4, 5, 6}]),
    "H11": (8, [{1, 3, 4, 5, 6}, {1, 2, 3, 5, 7}, {1, 4, 6, 7}, {2, 3, 4, 8}, {2, 3, 6, 7}, {1, 2, 6, 8}, {2, 4, 5, 7}, {3, 7, 8}, {5, 6, 8}, {1, 4, 5, 8}]),
    "H15": (8, "paired", [{3, 7, 8}, {4, 6, 8}, {2, 3, 5, 8}, {5, 6, 7}, {2, 3, 6, 7}, {2, 4, 5, 7}, {3, 4, 5, 6}]),
    "H17": (9, "paired", [{2, 3, 4, 8}, {3, 4, 5, 6}, {2, 4, 6, 7}, {2, 3, 5, 7}, {3, 4, 6, 8, 9}, {2, 4, 5, 7, 9}, {2, 3, 5, 6, 9}, {3, 4, 7, 9}]),
}



def star(radii, rho_O, J, beta):
    """The star inequality (6.3): radii of the N parties, radius of the purifier, RHS sets J_K (bitmasks) with coefficients beta_K."""
    N = len(radii)
    q = {1 << i: r for i, r in enumerate(radii) if r}
    q = add((1, q), (rho_O, {(1 << N) - 1: 1}), *[(-b, {m: 1}) for m, b in zip(J, beta)])
    return q, N


def coins(j):
    """Coin weights omega^(j) of (6.15) and the list of halves T_j of (6.16), coins numbered 0..m_j - 1."""
    w, halves = [1, 1, 1, 1], [{0, 1}, {0, 2}, {0, 3}]
    for _ in range(j):
        qj, u, v = sum(w), len(w), len(w) + 1
        halves = [T | {u} for T in halves] + [halves[0] | {v}, {u, v}]
        w = w + [qj // 2, qj // 2]
    return w, halves


def star_family_rhs(j):
    """D_j: for each half T in order a party with word 1_T and radius |T| and one with word 1_(complement of T) and radius m - |T|, then a party with word 1 and radius m - 1; purifier radius 1; beta = 1. Returns the radii and the RHS sets J_K in the order of the coins."""
    w, halves = coins(j)
    m = len(w)
    words = [x for T in halves for x in (T, set(range(m)) - T)] + [set(range(m))]
    radii = [len(x) for x in words[:-1]] + [m - 1]
    J = [sum(1 << i for i, x in enumerate(words) if K in x) for K in range(m)]
    return radii, J


def star_family(j):
    radii, J = star_family_rhs(j)
    return star(radii, 1, J, [1] * len(J))


D1 = star_family(1)[0]


def weighing_list(name):
    """The list (T_1, ..., T_(N-1)) of Theorem 6.14 for H9, ..., H17, with coins numbered 0..m-1."""
    m, *rest = WEIGHINGS[name]
    T = [{K - 1 for K in t} for t in rest[-1]]
    if rest[0] == "paired":
        T = T + [set(range(m)) - t for t in T]
    return m, T


def weighing_star(m, T):
    """The inequality of Theorem 6.14: party i < N - 1 with word 1_(T_i) and radius |T_i|, party N with word 1 and radius m - 1, purifier radius 1, beta = 1. Returns the inequality, N and the RHS sets J_K in the order of the coins."""
    words = [set(t) for t in T] + [set(range(m))]
    radii = [len(t) for t in T] + [m - 1]
    J = [sum(1 << i for i, x in enumerate(words) if K in x) for K in range(m)]
    q, N = star(radii, 1, J, [1] * m)
    return q, N, J


H = {name: weighing_star(*weighing_list(name)) for name in WEIGHINGS}

# name: (inequality, number of parties)
ALL = {"B7*": (B7STAR, 7), "B7": (B7, 7), "Q8A": (Q8A, 8), "Q8": (Q8, 8), "R8": (R8, 8), "Q8'": (Q8P, 8), "Q8''": (Q8PP, 8),
       **{f"E{s}": (E[s], 7) for s in range(6)}, **{f"mu{s}": (MU[s], 7) for s in range(6)},
       "F7": (F7, 7), "V7": (V7, 7), "V7'": (V7P, 7), "X7": (X7, 7), "Y7": (Y7, 7), "P7": (P7, 7), "(V7'+P7)/2": (HALF, 7),
       "Q'": (QPRIME, 3), "D0": (D0, 7), "D1": (D1, 11), "Z7": (Z7, 7), "U6": (U6, 6), "G9": (G9, 9), **{name: (w[0], w[1]) for name, w in H.items()}}

LISTING = Path(__file__).with_name("certificates") / "inequalities.txt"


def listing():
    return "".join(f"{name} (N = {N}): {text(q, N)}\n" for name, (q, N) in ALL.items())


def stab_order(q, N):
    return len(stabilizer(q, N))


def main():
    assert LISTING.read_text() == listing(), "certificates/inequalities.txt differs from the definitions"
    print("certificates/inequalities.txt agrees with the definitions")
    nm = names(7)
    pair = lambda i, j: nm[i] + nm[j]

    # Section 3: B7* and B7
    for name in ("B7*", "B7"):
        q, N = ALL[name]
        bv = bell_values(q, N)
        assert all(bv[(i, N)] == 0 for i in range(N)), name
        print(name, invariants(q, N), "balanced; Bell values on pairs of parties:", sorted(v for (i, j), v in bv.items() if j < N))
    bv = bell_values(B7STAR, 7)
    tri = {frozenset(p) for t in ("BCD", "EFG") for p in itertools.combinations(t, 2)}
    assert all(v == (4 if frozenset(pair(i, j)) in tri else 2) for (i, j), v in bv.items() if j < 7)
    assert invariants(B7STAR, 7) == {"N": 7, "L": 8, "R": 7, "|alpha|": 21, "|beta|": 20}
    lhs, _ = sides(B7STAR, 7)
    z = sum(1 << l for l, (m, _) in enumerate(lhs) if m != term_mask("A", 7))
    assert z in set(realizable_domain(lhs).tolist())
    print("B7*: Bell value 4 on the 6 pairs inside a triangle, 2 on the other 15; the point z of Prop. 3.2(iii) lies in the realizable domain")

    # Section 4.1: U6
    six = "ABCDEF"
    lhs6, rhs6 = sides(U6, 6)
    assert invariants(U6, 6) == {"N": 6, "L": 23, "R": 13, "|alpha|": 23, "|beta|": 33}
    pairs = {term_mask(a + b, 6) for a, b in itertools.combinations(six, 2)}
    fives = {term_mask(_other(six, x), 6) for x in six}
    assert {m for m, _ in lhs6} == pairs | {term_mask(t, 6) for t in TRIANGLES} | fives and all(c == 1 for _, c in lhs6)
    assert {m for m, c in rhs6 if bin(m).count("1") == 4} == {term_mask(e + "O", 6) for e in EDGES}
    bv = bell_values(U6, 6)
    assert len(bv) == 21 and all(v == 0 for v in bv.values())
    sU = stabilizer(U6, 6)
    tri_sets = {frozenset(term_mask(t, 6) for t in TRIANGLES)}
    keeps_triangles = lambda p: {frozenset(sum(1 << p[names(6).index(c)] for c in t) for t in TRIANGLES)} == tri_sets
    assert len(sU) == 72 and all(p[6] == 6 and keeps_triangles(p) for p in sU)
    assert [len(x) for x in PSI_INSTANCES] == [6, 2, 18, 18] and all(relabel(psi, p, 6) == psi for psi in PSI for p in sU)
    orbits = {frozenset(sum(1 << p[i] for i in range(6) if m >> i & 1) for p in sU) for m in subsets(6)}
    assert len(orbits) == 9
    print("U6:", invariants(U6, 6), "LHS = the 15 pairs, the two triangles and the six sets of five parties; the RHS terms with four parties are S(e O) for the six edges e; all 21 Bell values 0;"
          " |Stab| = 72, fixing the purifier and the pair of triangles; psi_1, ..., psi_4 are sums of 6, 2, 18, 18 instances of MMI, each invariant under Stab; Stab has 9 orbits on the 63 terms")
    dom6 = realizable_domain(lhs6)
    words6 = occurrence_words(lhs6, 6)
    z = sum(1 << l for l, (m, _) in enumerate(lhs6) if m in fives)
    d = lambda u, v: bin(u ^ v).count("1")
    assert dom6.size == 580 and z in set(dom6.tolist()) and all(d(z, w) == 7 for w in words6) and d(z, 0) == 6
    between = lambda u, v: (u & v) & ~z == 0 and z & ~(u | v) == 0
    assert not any(between(u, v) for u, v in itertools.combinations(words6 + [0], 2))
    # all integer values n(z) in the ranges against the seven pinned points (eq. (4.5)); pinned values are corners of the box, so |n_r - P_r| is n_r or beta_r - n_r
    beta6 = np.array([c for _, c in rhs6])
    pinned = [(np.array([c * (m >> i & 1) for m, c in rhs6], dtype=np.int8), 7) for i in range(6)] + [(np.zeros(13, dtype=np.int8), 6)]
    grid = np.stack(np.meshgrid(*[np.arange(b + 1, dtype=np.int8) for b in beta6], indexing="ij"), -1).reshape(-1, 13)
    ok = np.ones(len(grid), dtype=bool)
    for P, dist in pinned:
        ok &= np.abs(grid - P).sum(1) <= dist
    half = [Fraction(0)] * 6 + [Fraction(1, 2)] * 6 + [Fraction(3)]
    assert not ok.any() and all(sum(abs(x - int(y)) for x, y in zip(half, P)) == dist for P, dist in pinned)
    print(f"U6: |Real| = 580; the point z with the six sets of five parties lies in Real, d(z, x_i) = 7, d(z, 0) = 6, and lies between no two of the seven pinned points;"
          f" none of the {len(grid)} integer values of n(z) satisfies the contraction conditions (4.5) at the pinned points, while a = 0, b = 1/2, c = 3 satisfies all seven with equality")

    # Section 4.2: Q8A
    bv = bell_values(Q8A, 8)
    assert all(bv[(i, 8)] == 0 for i in range(8))
    vals = [v for (i, j), v in bv.items() if j < 8]
    assert sorted(vals) == [0] * 12 + [2] * 16 and all(m & 1 for m in Q8A)
    print("Q8A: every term contains A, balanced, 16 of the 28 pairs of parties have Bell value 2, the other 12 have 0")

    # Section 4.3
    assert invariants(Q8, 8) == {"N": 8, "L": 14, "R": 17, "|alpha|": 21, "|beta|": 41}
    assert all(v == 0 for v in bell_values(Q8, 8).values()) and len(bell_values(Q8, 8)) == 36
    assert stab_order(Q8, 8) == 8
    assert {m: c for m, c in Q8.items() if m & 1} == Q8A
    assert delete_party(Q8P, 7, 8) == E0 and delete_party(Q8PP, 7, 8) == E0
    print("Q8: L 14, R 17, |alpha| 21, |beta| 41, all 36 Bell values 0, |Stab| = 8; its terms containing A are those of Q8A; deleting H from Q8' and from Q8'' gives E0 of (4.8)")

    # Section 4.4: Fano structure of E0 and E1, stabilizers
    lines = ["ABC", "ADE", "AFG", "BDG", "BEF", "CDF", "CEG"]
    comp = lambda t: "".join(c for c in "ABCDEFG" if c not in t)
    e1_lhs = {term_mask(t, 7): (3 if "A" in t else 4) for t in lines}
    e1_lhs.update({term_mask(comp(t), 7): (3 if "A" not in t else 2) for t in lines})
    assert {m: c for m, c in E1.items() if c > 0} == e1_lhs
    sE0 = set(stabilizer(E0, 7))
    assert len(sE0) == 24 and stabilizer(E1, 7) == sorted(sE0) and all(p[0] == 0 and p[7] == 7 for p in sE0)
    print("E0, E1: |Stab| = 24, fixing A and the purifier; LHS of E1 as described in Section 4.4")
    sE4 = stabilizer(E4, 7)
    assert len(sE4) == 6 and all(p[7] == 7 for p in sE4)
    print("E4: |Stab| = 6, fixing the purifier (Section 6.1)")

    # Table 1
    table1 = {"Q8": (8, 14, 17, 21, 41, None), "Q8'": (8, 14, 17, 49, 91, None), "Q8''": (8, 14, 18, 49, 88, None),
              "E0": (7, 14, 16, 49, 83, 2187), "E1": (7, 14, 16, 43, 75, 2187), "E2": (7, 15, 19, 45, 75, 2916),
              "E3": (7, 15, 25, 42, 69, 2916), "E4": (7, 15, 24, 42, 66, 2916), "E5": (7, 18, 33, 48, 66, 6912)}
    print("Table 1: name N L R |alpha| |beta| |Real| |U| superbalanced antichain")
    for name, row in table1.items():
        q, N = ALL[name]
        lhs, _ = sides(q, N)
        inv = invariants(q, N)
        A = realizable_domain(lhs).size if N == 7 else None
        sb = all(v == 0 for v in bell_values(q, N).values())
        anti = antichain_at_A(lhs)
        U = u_domain(lhs, N).size
        got = (inv["N"], inv["L"], inv["R"], inv["|alpha|"], inv["|beta|"], A)
        assert got == row and sb and anti and U == (135 if N == 8 else 134), (name, got)
        print(" ", name, *got, U, sb, anti)

    # Section 5.1: F7
    assert invariants(F7, 7) == {"N": 7, "L": 20, "R": 24, "|alpha|": 39, "|beta|": 54}
    assert max(abs(c) for c in F7.values()) == 4 and gcd(*F7.values()) == 1
    assert len(bell_values(F7, 7)) == 28 and all(v == 0 for v in bell_values(F7, 7).values())
    lhsF, _ = sides(F7, 7)
    assert {m: c for m, c in F7.items() if m & 1 and c > 0} == {m: 2 for m, c in E0.items() if m & 1 and c > 0} and antichain_at_A(lhsF)
    sF = stabilizer(F7, 7)
    gen = {tuple(permutation("BD,CE", 7)), tuple(permutation("BF,CG", 7))}
    closure = set(gen) | {tuple(range(8))}
    while True:
        new = {tuple(a[b[i]] for i in range(8)) for a in closure for b in closure} | closure
        if new == closure:
            break
        closure = new
    assert sorted(closure) == sF and len(sF) == 6 and set(sF) <= sE0 and all(p[0] == 0 and p[7] == 7 for p in sF)
    assert factorial(8) // len(sF) == 6720 and merge_pair(F7, 7) is None
    assert realizable_domain(lhsF).size == 11046
    print("F7: L 20, R 24, |alpha| 39, |beta| 54, coefficients up to 4 with gcd 1, 28 Bell values 0, A-terms those of E0 with weight 2 (an antichain),"
          " |Real| = 11046 of 2^20, Stab of order 6 generated by (BD)(CE) and (BF)(CG), inside Stab(E0), orbit 8!/6 = 6720, no pair of points always together")

    # Section 5.3 and Table 2
    for name in ("F7", "X7", "Y7"):
        assert tuple(G) in stabilizer(ALL[name][0], 7)
    assert stabilizer(F7, 7) == stabilizer(X7, 7) == stabilizer(Y7, 7)
    assert tuple(G) not in stabilizer(V7, 7)
    assert stabilizer(V7, 7) == sorted({tuple(range(8)), tuple(permutation("BF,CG", 7))}) and len(stabilizer(V7P, 7)) == 1
    table2 = {"F7": (20, 24, 39, 54, 6, 11046), "V7": (20, 26, 35, 48, 2, 10032), "V7'": (20, 23, 32, 44, 1, 10032),
              "X7": (18, 33, 102, 138, 6, 6912), "Y7": (21, 27, 66, 93, 6, 12774), "P7": (18, 35, 60, 80, 8, 6912)}
    print("Table 2: name L R |alpha| |beta| |Stab| |Real| superbalanced")
    for name, row in table2.items():
        q, N = ALL[name]
        inv = invariants(q, N)
        got = (inv["L"], inv["R"], inv["|alpha|"], inv["|beta|"], stab_order(q, N), realizable_domain(sides(q, N)[0]).size)
        sb = all(v == 0 for v in bell_values(q, N).values())
        assert got == row and sb, (name, got)
        print(" ", name, *got, sb)
    print("g = (DF)(EG) lies in Stab(F7) = Stab(X7) = Stab(Y7) and not in Stab(V7); Stab(V7) = {1, (BF)(CG)}; Stab(V7') is trivial")

    # Remark 5.9: the half-sum
    lhsH, _ = sides(HALF, 7)
    assert all(v == 0 for v in bell_values(HALF, 7).values())
    assert {m for m, c in HALF.items() if m & 1 and c > 0} == {m for m, c in E0.items() if m & 1 and c > 0} and antichain_at_A(lhsH) and u_domain(lhsH, 7).size == 134
    print("(V7' + P7)/2: an integer vector, superbalanced,", invariants(HALF, 7), "; its LHS terms containing A are those of E0 and form an antichain; |U| = 134")

    # Section 6: stars written out
    assert star_family(0) == (D0, 7)
    print("D0 of (6.18) is the star inequality D_0 of Section 6.3 with the parties named A, ..., G in the order of the construction")
    for name in ("Q'", "D0", "D1", "Z7", "G9", "H9", "H11", "H15", "H17"):
        q, N = ALL[name]
        print(name, invariants(q, N))
    return 0


if __name__ == "__main__":
    sys.exit(main())
