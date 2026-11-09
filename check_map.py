"""Exact checker for the maps of Appendix A.2: real (rational) and integer maps for kQ on the full cube or on the realizable domain.

A certificate in certificates/maps/ names the inequality, the scaling k and the domain, lists the LHS and RHS terms, and gives the values n_r(u) at every point u (a bitmask over the LHS terms). The checker works from the inequality as defined in inequalities.py: it verifies that the certificate's terms are those of the inequality, rebuilds the domain from Definition 2.11 (or takes the full cube) and verifies that the set of points of the certificate is exactly this domain, verifies the pins (2.7) and the ranges 0 <= n_r <= k beta_r, and verifies the contraction condition (2.8) in exact arithmetic on every pair of points of the realizable domain, or on every edge of the full cube (enough by Lemma 2.12) and, for cubes of at most 256 points, also on every pair. It also reports whether the map is equivariant under the relabelings in Stab(Q) that fix the purifier.

Usage: python check_map.py [certificate ...]   (default: every certificate in certificates/maps/)"""
import sys
from fractions import Fraction
from math import lcm
from pathlib import Path

import numpy as np

import inequalities as I
from core import occurrence_words, read_map, realizable_domain, sides, stabilizer, term_name

MAPS = Path(__file__).with_name("certificates") / "maps"


def load(path):
    cert = read_map(path)
    q, N = I.ALL[cert["inequality"]]
    k = cert["k"]
    lhs, rhs = sides(q, N)
    assert cert["lhs"] == [term_name(m, N) for m, _ in lhs] and cert["rhs"] == [term_name(m, N) for m, _ in rhs], "terms differ from the inequality"
    alpha = np.array([k * c for _, c in lhs], dtype=np.int64)
    beta = [k * c for _, c in rhs]
    vals = [[Fraction(v) for v in row] for row in cert["values"]]
    D = 1
    for row in vals:
        for v in row:
            D = lcm(D, v.denominator)
    order = np.argsort(cert["points"])
    pts = np.array(cert["points"], dtype=np.int64)[order]
    num = [[int(v * D) for v in vals[i]] for i in order]
    return cert, q, N, lhs, rhs, alpha, beta, pts, num, D


def check_pairs(pts, num, D, alpha, beta, L, pairs):
    """Contraction on the given pairs, or on all pairs if pairs is None; returns (pairs checked, tight pairs, violations)."""
    R = len(beta)
    bits = ((pts[:, None] >> np.arange(L)) & 1).astype(np.int8)
    if D == 1 and max(beta) < 2**31:
        V = np.array(num, dtype=np.int64)
        exact = None
    else:
        V = np.array([[x / D for x in row] for row in num])
        exact = num
        # A Fraction converts to the nearest float, so each value is off by at most max(beta) * 2^-53; differences, absolute values and a sum of R terms add at most (3R + R^2) such errors. Pairs whose float slack exceeds this bound hold; the rest are decided in exact integer arithmetic.
        tol = (3 * R + R * R) * max(beta) * 2.0**-52
    n = tight = viol = 0

    def batch(i, js):
        nonlocal n, tight, viol
        dist = (bits[js] != bits[i]) @ alpha
        if exact is None:
            s = np.abs(V[js] - V[i]).sum(1)
            viol += int((s > dist).sum())
            tight += int((s == dist).sum())
        else:
            slack = dist - np.abs(V[js] - V[i]).sum(1)
            for pos in np.flatnonzero(slack <= tol):
                s = sum(abs(a - b) for a, b in zip(exact[js[pos]], exact[i]))
                d = int(dist[pos]) * D
                viol += s > d
                tight += s == d
        n += js.size

    if pairs is None:
        for i in range(pts.size - 1):
            batch(i, np.arange(i + 1, pts.size))
    else:
        for i, js in pairs:
            batch(i, js)
    return n, tight, viol


def cube_edges(L):
    """For each point x with bit l clear, the pair (x, x + 2^l); grouped by x."""
    out = []
    for x in range(1 << L):
        js = np.array([x | (1 << l) for l in range(L) if not x >> l & 1], dtype=np.int64)
        if js.size:
            out.append((x, js))
    return out


def equivariant(q, N, lhs, rhs, pts, num):
    index = {int(x): i for i, x in enumerate(pts)}
    L = len(lhs)
    for p in stabilizer(q, N):
        if p[N] != N:
            continue
        img = lambda m: sum(1 << p[i] for i in range(N) if m >> i & 1)
        lp = [[m for m, _ in lhs].index(img(m)) for m, _ in lhs]
        rp = [[m for m, _ in rhs].index(img(m)) for m, _ in rhs]
        for x, i in index.items():
            gx = sum(1 << lp[l] for l in range(L) if x >> l & 1)
            if any(num[index[gx]][rp[r]] != num[i][r] for r in range(len(rhs))):
                return False
    return True


def check(path):
    cert, q, N, lhs, rhs, alpha, beta, pts, num, D = load(path)
    L, R = len(lhs), len(rhs)
    k, dom = cert["k"], cert["domain"]
    want = realizable_domain(lhs) if dom == "realizable" else np.arange(1 << L, dtype=np.int64)
    assert np.array_equal(pts, want), "the points of the certificate are not the domain"
    index = {int(x): i for i, x in enumerate(pts)}
    words = occurrence_words(lhs, N)
    for i, w in enumerate(words):
        assert num[index[w]] == [beta[r] * (rhs[r][0] >> i & 1) * D for r in range(R)], f"pin of party {i}"
    assert num[index[0]] == [0] * R, "pin of the purifier"
    assert all(0 <= row[r] <= beta[r] * D for row in num for r in range(R)), "range"
    kind = "integer" if D == 1 else f"rational, common denominator with {len(str(D))} digits"
    print(f"{path.name}: {'' if k == 1 else k}{cert['inequality']}, {kind}, domain {dom} ({pts.size} points), terms, domain, pins and ranges ok", flush=True)
    if dom == "realizable":
        n, t, v = check_pairs(pts, num, D, alpha, beta, L, None)
        print(f"  all pairs: {n} checked, {t} tight, {v} violations", flush=True)
    else:
        n, t, v = check_pairs(pts, num, D, alpha, beta, L, cube_edges(L))
        print(f"  all edges: {n} checked, {t} tight, {v} violations", flush=True)
        if pts.size <= 256:
            n_all, t_all, v_all = check_pairs(pts, num, D, alpha, beta, L, None)
            print(f"  all pairs: {n_all} checked, {t_all} tight, {v_all} violations", flush=True)
            v += v_all
    if dom == "realizable":
        print(f"  equivariant under the relabelings in Stab({cert['inequality']}) that fix the purifier: {equivariant(q, N, lhs, rhs, pts, num)}", flush=True)
    assert v == 0, "contraction condition violated"
    return True


def main(args):
    paths = [Path(a) for a in args] or sorted(MAPS.glob("*.tsv"))
    for p in paths:
        check(p)
    print(f"{len(paths)} map(s): all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
