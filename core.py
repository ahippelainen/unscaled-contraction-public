"""Definitions shared by the scripts: entropy inequalities as coefficient vectors, purification, occurrence words, Bell values, relabelings and stabilizers, the realizable domain, the subdomain U, the domain of the star certificates, irreducible pairs, and instances of subadditivity and MMI.

Parties are A, B, C, ... (the letter O is skipped), and O is the purifier. A set of parties is a bitmask over the N parties; a set that contains the purifier is replaced by its complement in the parties and the purifier. An inequality sum_l alpha_l S(I_l) >= sum_r beta_r S(J_r) is stored as a dict {mask: coefficient}, positive on the left-hand side and negative on the right, so that it reads Q . S >= 0. Vectors are indexed by the nonempty subsets of the parties ordered by size and then lexicographically."""
import itertools
import re
from functools import lru_cache
from math import gcd

import numpy as np

LETTERS = [c for c in "ABCDEFGHIJKLMNPQRSTUVWXYZ"]


def names(N):
    return LETTERS[:N] + ["O"]


def term_mask(word, N):
    """Mask of a term such as 'ABO', purified."""
    nm = names(N)
    m = 0
    for ch in word:
        m |= 1 << nm.index(ch)
    if m >> N & 1:
        m ^= (1 << (N + 1)) - 1
    return m


def term_name(m, N):
    return "".join(LETTERS[i] for i in range(N) if m >> i & 1)


def parse(text, N):
    """Parse '2A + AO + 3(BO + CO) >= ABC + 14O' into a coefficient dict."""
    def side(s, sign, out):
        s = s.replace(" ", "")
        pos = 0
        for m in re.finditer(r"(\d*)\(([^)]*)\)|(\d*)([A-Z]+)", s):
            assert m.start() == pos or s[pos:m.start()] == "+", s
            pos = m.end()
            if m.group(2) is not None:
                c = int(m.group(1) or 1)
                side_terms = m.group(2).split("+")
            else:
                c = int(m.group(3) or 1)
                side_terms = [m.group(4)]
            for t in side_terms:
                k = term_mask(t, N)
                assert k, t
                out[k] = out.get(k, 0) + sign * c
        assert pos == len(s), s
    lhs, rhs = text.split(">=")
    q = {}
    side(lhs, 1, q)
    side(rhs, -1, q)
    return {m: c for m, c in q.items() if c}


@lru_cache(None)
def subsets(N):
    return [sum(1 << i for i in c) for s in range(1, N + 1) for c in itertools.combinations(range(N), s)]


def vector(q, N):
    pos = {m: i for i, m in enumerate(subsets(N))}
    v = np.zeros(len(pos), dtype=object)
    for m, c in q.items():
        v[pos[m]] += c
    return v


def add(*pairs):
    """Linear combination sum c_i q_i of coefficient dicts, given as (c_i, q_i)."""
    out = {}
    for c, q in pairs:
        for m, x in q.items():
            out[m] = out.get(m, 0) + c * x
    return {m: x for m, x in out.items() if x}


def sides(q, N):
    """LHS and RHS as lists of (mask, coefficient), in vector order."""
    order = subsets(N)
    lhs = [(m, q[m]) for m in order if q.get(m, 0) > 0]
    rhs = [(m, -q[m]) for m in order if q.get(m, 0) < 0]
    return lhs, rhs


def text(q, N):
    """The inequality written out, terms on each side ordered by size and then lexicographically."""
    lhs, rhs = sides(q, N)
    fmt = lambda side: " + ".join((str(c) if c != 1 else "") + term_name(m, N) for m, c in side)
    return fmt(lhs) + " >= " + fmt(rhs)


def invariants(q, N):
    lhs, rhs = sides(q, N)
    return {"N": N, "L": len(lhs), "R": len(rhs), "|alpha|": sum(c for _, c in lhs), "|beta|": sum(c for _, c in rhs)}


def separates(m, i, j):
    """Whether the purified term m contains exactly one of the points i, j (index N = purifier, never in m)."""
    return (m >> i & 1) != (m >> j & 1)


def bell(q, i, j):
    """Q . r_ij for distinct points i, j of the N parties and the purifier (index N)."""
    return sum(c for m, c in q.items() if separates(m, i, j))


def bell_values(q, N):
    return {(i, j): bell(q, i, j) for i, j in itertools.combinations(range(N + 1), 2)}


def relabel(q, perm, N):
    """Image of q under the permutation perm of the N + 1 points (perm[i] = image of point i, the purifier is point N)."""
    full = (1 << (N + 1)) - 1
    out = {}
    for m, c in q.items():
        img = sum(1 << perm[i] for i in range(N + 1) if m >> i & 1)
        if img >> N & 1:
            img ^= full
        out[img] = out.get(img, 0) + c
    return out


def permutation(cycles, N):
    """Permutation of the N + 1 points from cycles written with point names, e.g. 'DF,EG'."""
    nm = names(N)
    perm = list(range(N + 1))
    for cyc in filter(None, cycles.split(",")):
        idx = [nm.index(ch) for ch in cyc]
        for a, b in zip(idx, idx[1:] + idx[:1]):
            perm[a] = b
    return perm


@lru_cache(None)
def _perm_table(N):
    """For every permutation p of the N + 1 points, the index of the image of every subset (with purification)."""
    order = subsets(N)
    full = (1 << (N + 1)) - 1
    perms = np.array(list(itertools.permutations(range(N + 1))), dtype=np.int16)
    bits = np.array([[(m >> i) & 1 for i in range(N + 1)] for m in order], dtype=np.int16)
    img = np.zeros((len(perms), len(order)), dtype=np.int16)
    for i in range(N + 1):
        img |= bits[None, :, i] << perms[:, i][:, None]
    img = np.where(img >> N & 1, full ^ img, img)
    lut = np.zeros(full + 1, dtype=np.int16)
    lut[order] = np.arange(len(order))
    return perms, lut[img]


def stabilizer(q, N):
    """All permutations of the N + 1 points that fix the coefficient vector of q."""
    perms, PT = _perm_table(N)
    v = np.array(vector(q, N), dtype=np.int64)
    keep = np.ones(len(perms), dtype=bool)
    for k in np.flatnonzero(v):
        keep &= v[PT[:, k]] == v[k]
    return [tuple(int(x) for x in p) for p in perms[keep]]


def canonical_form(q, N):
    """Lexicographically smallest image of q / gcd under the (N+1)! relabelings, and the orbit size."""
    perms, PT = _perm_table(N)
    v = np.array(vector(q, N), dtype=np.int64)
    v //= gcd(*map(int, np.abs(v[v != 0])))
    V = np.zeros(PT.shape, dtype=np.int64)
    np.put_along_axis(V, PT.astype(np.int64), np.broadcast_to(v, PT.shape), axis=1)
    U = np.unique(V, axis=0)
    return U[0], len(U)


def merge_pair(q, N):
    """A pair of the N + 1 points such that every term contains both or neither, or None."""
    for a, b in itertools.combinations(range(N + 1), 2):
        if all(not separates(m, a, b) for m in q):
            return a, b
    return None


def delete_party(q, i, N):
    """Replace every term S(I) by S(I minus {i}); returns an inequality on the other N - 1 parties."""
    out = {}
    for m, c in q.items():
        m2 = m & ~(1 << i)
        m2 = (m2 & ((1 << i) - 1)) | ((m2 >> (i + 1)) << i)
        if m2:
            out[m2] = out.get(m2, 0) + c
    return {m: c for m, c in out.items() if c}


def sa(X, Y, N):
    """S(X) + S(Y) >= S(XY) for disjoint nonempty sets of points (strings, may contain O)."""
    return add(*[(c, {term_mask(t, N): 1}) for c, t in ((1, X), (1, Y), (-1, X + Y)) if term_mask(t, N)])


def mmi(X, Y, Z, N):
    """S(XY) + S(XZ) + S(YZ) >= S(X) + S(Y) + S(Z) + S(XYZ) for disjoint nonempty sets of points (strings, may contain O)."""
    pos = [X + Y, X + Z, Y + Z]
    neg = [X, Y, Z, X + Y + Z]
    return add(*[(1, {term_mask(t, N): 1}) for t in pos], *[(-1, {term_mask(t, N): 1}) for t in neg if term_mask(t, N)])


def occurrence_words(lhs, N):
    """LHS occurrence words x^(i) as bitmasks over the L LHS terms, for the parties i = 0..N-1."""
    return [sum(1 << l for l, (m, _) in enumerate(lhs) if m >> i & 1) for i in range(N)]


def realizable_domain(lhs):
    """All x in {0,1}^L (as integers) with (R1) x_l = x_l' = 1 => I_l, I_l' intersect, and (R2) x_l = 1, I_l subset of I_l' => x_l' = 1."""
    L = len(lhs)
    X = np.arange(1 << L, dtype=np.int64)
    ok = np.ones(X.size, dtype=bool)
    for l in range(L):
        xl = (X >> l) & 1
        for m in range(L):
            if m == l:
                continue
            xm = (X >> m) & 1
            if lhs[l][0] & lhs[m][0] == 0:
                ok &= ~((xl == 1) & (xm == 1))
            if lhs[l][0] & ~lhs[m][0] == 0:
                ok &= ~((xl == 1) & (xm == 0))
    return X[ok]


def u_domain(lhs, N):
    """U(Q): the occurrence words, the word 0, and the interval [0, x^(A)] (party A = index 0)."""
    words = occurrence_words(lhs, N)
    xa = words[0]
    sub = [x for x in range(xa + 1) if x & ~xa == 0]
    return np.array(sorted(set(words) | {0} | set(sub)), dtype=np.int64)


def star_domain(lhs, N):
    """For a star inequality with all radii positive: the N + 2 points 0, z (only the bit of S(parties) set) and the occurrence words (Appendix A.1)."""
    full = (1 << N) - 1
    assert [m for m, _ in lhs] == [1 << i for i in range(N)] + [full], "not a star inequality with all radii positive"
    z = 1 << N
    return np.array(sorted({0, z} | set(occurrence_words(lhs, N))), dtype=np.int64)


def antichain_at_A(lhs):
    """Whether the LHS terms that contain party A are pairwise non-nested."""
    ta = [m for m, _ in lhs if m & 1]
    return all(not (a & ~b == 0 or b & ~a == 0) for a, b in itertools.combinations(ta, 2))


def irreducible_partners(dom, xi, L):
    """Indices j of the points y of dom (sorted array of integers) such that no third point of dom lies in the box between x = dom[xi] and y: the difference y ^ x must be minimal under inclusion among the differences z ^ x, z in dom, z != x."""
    x = int(dom[xi])
    D = dom ^ x
    has = np.zeros(1 << L, dtype=bool)
    has[D[D != 0]] = True
    for b in range(L):
        h = has.reshape(-1, 2, 1 << b)
        h[:, 1, :] |= h[:, 0, :]
    below = np.zeros(dom.size, dtype=bool)
    for b in range(L):
        bit = (D >> b) & 1 == 1
        below[bit] |= has[D[bit] ^ (1 << b)]
    return np.flatnonzero((D != 0) & ~below)


DOMAIN_NAMES = {"cube": "the full cube", "realizable": "the realizable domain"}


def write_map(path, cert):
    """Write a map certificate as a tab-separated table: a '#' header with the inequality, the scaling k, the domain and the LHS and RHS terms, then one row per point, the point as its 0/1 word over the LHS terms in the listed order, followed by one value per RHS term (integers, or fractions a/b)."""
    lhs, rhs = cert["lhs"], cert["rhs"]
    L = len(lhs)
    rows = sorted(zip(cert["points"], cert["values"]))
    with open(path, "w") as fh:
        fh.write(f"# Map for {cert['k']}*{cert['inequality']} on {DOMAIN_NAMES.get(cert['domain'], cert['domain'])}, {len(rows)} points\n")
        fh.write(f"# inequality: {cert['inequality']}\n# k: {cert['k']}\n# domain: {cert['domain']}\n")
        fh.write("# lhs: " + " ".join(lhs) + "\n# rhs: " + " ".join(rhs) + "\n")
        fh.write("point\t" + "\t".join(rhs) + "\n")
        for x, vals in rows:
            fh.write("".join("1" if x >> l & 1 else "0" for l in range(L)) + "\t" + "\t".join(str(v) for v in vals) + "\n")


def read_map(path):
    """Read a map certificate written by write_map; points are returned as integers with bit l for the l-th LHS term."""
    cert, points, values = {}, [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                key, sep, val = line[1:].strip().partition(": ")
                if sep and key in ("inequality", "k", "domain", "lhs", "rhs"):
                    cert[key] = int(val) if key == "k" else val.split(" ") if key in ("lhs", "rhs") else val
            elif line.startswith("point\t"):
                assert line.rstrip("\n").split("\t")[1:] == cert["rhs"], "column header differs from the rhs line"
            else:
                word, *vals = line.rstrip("\n").split("\t")
                points.append(sum(1 << l for l, b in enumerate(word) if b == "1"))
                values.append(vals)
    cert["points"], cert["values"] = points, values
    return cert
