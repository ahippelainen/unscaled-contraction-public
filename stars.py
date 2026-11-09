"""Star inequalities (Sections 6.2-6.5, Table 3, Appendix C.2): the star polytope of Definition 6.2, a proof in exact rational arithmetic that it is a single point for Q', D_0, D_1, D_2, Z_7, G_9, H_9, H_11, H_15 and H_17, the identities of Lemma 6.8, Theorem 6.14(b) and Theorem 6.17(b), the conditions (W1) and (W2) of Theorem 6.14 for the lists of Appendix C.2, the dependences of Example 6.19 and of G_9, and the Bell values of Section 6.5. The integer maps at k = k_min of Table 3 are certificates in certificates/maps/, checked by check_map.py.

That the polytope is the single point c is certified coordinate by coordinate: for each K and each sign, nonnegative rational multipliers of the constraint rows whose combination is +e_K or -e_K bound +c_K or -c_K from above (Farkas); the multipliers come from a linear program in floating point and are then rationalized and checked exactly, and the upper and lower bounds must coincide."""
import sys
from fractions import Fraction
from math import lcm

import numpy as np
from scipy.optimize import linprog

import inequalities as I
from core import add, bell, term_mask


def star_data(q, N, J):
    """Radii of the parties and of the purifier and RHS coefficients of a star inequality whose RHS sets are J (in this order)."""
    full = (1 << N) - 1
    lhs_ok = all(c > 0 for m, c in q.items() if bin(m).count("1") == 1 or m == full)
    assert lhs_ok and sorted(m for m, c in q.items() if c < 0) == sorted(J), "not a star inequality with these RHS terms"
    assert all(2 <= bin(m).count("1") <= N - 1 for m in J) and len(set(J)) == len(J)
    return [q.get(1 << i, 0) for i in range(N)], q.get(full, 0), [-q[m] for m in J]


def constraints(N, J, radii, rho_O, beta):
    """||c - eta^(i)||_1 <= rho_i for the parties and the purifier, linear on the box by Lemma 6.5, as rows (a, b) meaning a . c <= b; then the box."""
    m = len(J)
    radial = []
    for i in range(N + 1):
        inside = [i < N and bool(J[K] >> i & 1) for K in range(m)]
        radial.append(([-1 if x else 1 for x in inside], Fraction((radii[i] if i < N else rho_O) - sum(b for b, x in zip(beta, inside) if x))))
    box = [([int(j == K) for j in range(m)], Fraction(beta[K])) for K in range(m)] + [([-int(j == K) for j in range(m)], Fraction(0)) for K in range(m)]
    return radial, box


def single_point(cons):
    """Certify that {c : a . c <= b for all rows} is one point and return it."""
    m = len(cons[0][0])
    A = np.array([a for a, _ in cons], dtype=float)
    b = np.array([float(x) for _, x in cons])
    c = []
    for K in range(m):
        bounds = []
        for sgn in (1, -1):
            obj = np.zeros(m)
            obj[K] = -sgn
            res = linprog(obj, A_ub=A, b_ub=b, bounds=[(None, None)] * m, method="highs")
            assert res.status == 0
            lam = [Fraction(-x).limit_denominator(10**6) for x in res.ineqlin.marginals]
            assert all(x >= 0 for x in lam)
            assert [sum(l * a[j] for l, (a, _) in zip(lam, cons)) for j in range(m)] == [sgn * int(j == K) for j in range(m)], "multipliers not exact"
            bounds.append(sgn * sum(l * be for l, (_, be) in zip(lam, cons)))
        assert bounds[0] == bounds[1], (K, bounds)
        c.append(bounds[0])
    assert all(sum(Fraction(x) * y for x, y in zip(a, c)) <= be for a, be in cons)
    return c


def chains(q, N, J, beta, c):
    """Lemma 6.8 at c times T = lcm of the denominators: T Q minus sum_K [(T beta_K - T c_K) SA1_K + T c_K SA2_K]; must be a nonnegative combination of positivity of single parties and of S(parties)."""
    T = lcm(*[x.denominator for x in c])
    full = (1 << N) - 1
    total = {}
    for K, msk in enumerate(J):
        sa1 = add(*[(1, {1 << i: 1}) for i in range(N) if msk >> i & 1], (-1, {msk: 1}))
        sa2 = add(*[(1, {1 << i: 1}) for i in range(N) if not msk >> i & 1], (1, {full: 1}), (-1, {msk: 1}))
        total = add((1, total), (T * beta[K] - T * c[K], sa1), (T * c[K], sa2))
    rest = add((T, q), (-1, total))
    assert all(v > 0 and (bin(m).count("1") == 1 or m == full) for m, v in rest.items())
    return T, rest


def rank(rows):
    """Rank of an integer matrix, by elimination in exact rational arithmetic."""
    A = [[Fraction(x) for x in r] for r in rows]
    r = 0
    for c in range(len(A[0])):
        piv = next((i for i in range(r, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c] / A[r][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        r += 1
    return r


def positive_dependence(radial, c):
    """All radii attained at c, and the constraint rows have rank m and a linear dependence with positive coefficients (Example 6.19)."""
    m = len(c)
    assert all(sum(a[K] * c[K] for K in range(m)) == b for a, b in radial), "a radius is not attained"
    assert rank([a for a, _ in radial]) == m
    M = np.array([a for a, _ in radial], dtype=float)
    null = np.linalg.svd(M.T)[2][-1]
    lam = [Fraction(x).limit_denominator(1000) for x in null / null[np.argmax(np.abs(null))]]
    assert all(x > 0 for x in lam) and all(sum(l * a[K] for l, (a, _) in zip(lam, radial)) == 0 for K in range(m))
    return lam


def weighing_conditions(name, J, c):
    """(W1) and (W2) of Theorem 6.14 for the list of Appendix C.2, with omega = q c."""
    m, T = I.weighing_list(name)
    q = lcm(*[x.denominator for x in c])
    omega = [int(q * x) for x in c]
    rows = [[-1 if K in t else 1 for K in range(m)] for t in T]
    assert all(sum(r[K] * omega[K] for K in range(m)) == 0 for r in rows) and rank(rows) == m - 1 and all(w > 0 for w in omega) and sum(omega) == q
    assert all(sum(r[K] for r in rows) == 0 for K in range(m))
    J_expected = [sum(1 << i for i, t in enumerate(T) if K in t) | (1 << len(T)) for K in range(m)]
    assert J == J_expected
    return q, omega, rows


TABLE3 = {"D0": (7, 4, [1, 1, 1, 1], 4), "D1": (11, 6, [1, 1, 1, 1, 2, 2], 8), "D2": (15, 8, [1, 1, 1, 1, 2, 2, 4, 4], 16), "Z7": (7, 7, [2, 1, 1, 6, 3, 1, 2], 8),
          "G9": (9, 9, [2, 25, 28, 31, 10, 8, 1, 19, 4], 32), "H9": (9, 8, [1, 1, 2, 2, 2, 3, 3, 4], 18), "H11": (11, 8, [1, 2, 2, 3, 3, 4, 5, 6], 26),
          "H15": (15, 8, [1, 2, 3, 4, 5, 6, 7, 8], 36), "H17": (17, 9, [6, 4, 6, 5, 2, 3, 4, 1, 1], 32)}


def main():
    cases = [("Q'", I.QPRIME, 3, [term_mask(t, 3) for t in ("AB", "AC")], [Fraction(1, 2)] * 2)]
    for j in range(3):
        q, N = I.star_family(j)
        w, _ = I.coins(j)
        assert N == 4 * j + 7 and len(w) == 2 * j + 4 and sum(w) == 2 ** (j + 2) and w[0] == 1
        cases.append((f"D{j}", q, N, I.star_family_rhs(j)[1], [Fraction(x, sum(w)) for x in w]))
    cases.append(("Z7", I.Z7, 7, [term_mask(t, 7) for t in ("ABCG", "ABDE", "ACEF", "BCEF", "BEFG", "CDEG", "CDFG")], [Fraction(x, 8) for x in (2, 1, 1, 6, 3, 1, 2)]))
    cases.append(("G9", I.G9, 9, [term_mask(t, 9) for t in ("ABCDG", "ABCHI", "ABEFG", "ACDEF", "ACEGH", "BCDEH", "BCFGH", "CDEGI", "DFGHI")], [Fraction(x, 32) for x in TABLE3["G9"][2]]))
    for name in ("H9", "H11", "H15", "H17"):
        q, N, J = I.H[name]
        cases.append((name, q, N, J, [Fraction(x, TABLE3[name][3]) for x in TABLE3[name][2]]))
    for name, q, N, J, expected in cases:
        radii, rho_O, beta = star_data(q, N, J)
        radial, box = constraints(N, J, radii, rho_O, beta)
        c = single_point(radial + box)
        assert c == expected
        T, rest = chains(q, N, J, beta, c)
        unbalanced = sum(bell(q, i, N) != 0 for i in range(N))
        assert unbalanced
        print(f"{name}: N = {N}, m = {len(J)}, the star polytope is the single point c = {[str(x) for x in c]}, least common denominator {lcm(*[x.denominator for x in c])}")
        print(f"  {T} {name} = sum over K of [(T beta_K - T c_K) SA1_K + T c_K SA2_K]" + (" exactly" if not rest else " + positivity terms"))
        print(f"  {unbalanced} of the {N} parties have a nonzero Bell value with the purifier, so {name} is not balanced")
        if name in TABLE3:
            assert (N, len(J), [int(x * TABLE3[name][3]) for x in c], lcm(*[x.denominator for x in c])) == (TABLE3[name][0], TABLE3[name][1], TABLE3[name][2], TABLE3[name][3]), name
        if name in ("Z7", "G9"):
            words = [sum(1 << K for K in range(len(J)) if J[K] >> i & 1) for i in range(N)] + [0]
            full = (1 << len(J)) - 1
            assert not any(w ^ v == full for w in words for v in words), "complementary pair of words"
            assert not rest
            lam = positive_dependence(radial, c)
            print(f"  no complementary pair of words; all {N + 1} radii are attained at c, and the {N + 1} constraint rows have rank {len(c)} and the positive dependence {[str(x) for x in lam]}")
        if name.startswith("H"):
            qw, omega, rows = weighing_conditions(name, J, c)
            assert not rest
            kind = "half-regular (every coin in exactly half of the sets)" if name in ("H9", "H11") else "paired"
            if name in ("H9", "H11"):
                assert all(sum(K in t for t in I.weighing_list(name)[1]) * 2 == len(rows) for K in range(len(J)))
            print(f"  Theorem 6.14: (W1) the balanced vectors form the line spanned by omega = {omega} (rank {len(J) - 1}), (W2) with all gamma_i = 1, the list is {kind}; q = {qw}")
            assert [bell(q, i, N) for i in range(N)] == [1] * (N - 1) + [0]
            print(f"  Q . r_iO = 1 for the {N - 1} weighing parties and 0 for the last party")
        if name.startswith("D"):
            j = int(name[1])
            assert [bell(q, i, N) for i in range(N)] == [1] * (4 * j + 6) + [0]
            print(f"  Q . r_iO = 1 for the {4 * j + 6} parties with a word 1_T or its complement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
