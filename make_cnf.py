"""CNF encoding of "kQ has an integer map on the domain Omega" (Appendix A.1), optionally restricted to maps equivariant under Stab(Q) (Appendix A.4).

Points: Omega is the full cube {0,1}^L, the subdomain U(Q) of (2.14), the realizable domain of Definition 2.11, or, for star inequalities, the N + 2 points 0, z and the occurrence words of Appendix A.1, rebuilt here from the definitions. Values: for each point u and RHS term r an integer n_r(u) in [0, k beta_r], in the order encoding, by the variables [n_r(u) >= j], j = 1, ..., k beta_r, with [n_r(u) >= j+1] => [n_r(u) >= j]. Pins (2.7): unit clauses. Contraction (2.8): one pair from each class of pairs irreducible in Omega (Lemma 2.12), pairs with k d_alpha(u, v) >= k|beta| dropped; for a pair, one auxiliary variable for each order variable marks a difference between u and v (at least the exclusive or of the two), and a cardinality network allows at most k d_alpha(u, v) of them to be true. With --sym, the group G of relabelings that fix q (they fix the purifier for the inequalities of the paper) acts on points and terms, the variables of (u, r, j) and (g u, g(r), j) are identified, and the contraction condition is imposed on one pair from each G-orbit of irreducible pairs.

Every map (every equivariant map, with --sym) gives a satisfying assignment, since dropping pairs, using orbit representatives and the one-sided auxiliary variables only weaken the formula; so if the CNF is unsatisfiable, no such map exists. Unsatisfiability is certified by a DRAT proof: write one with a proof-producing solver and check it with drat-trim (see README).

Usage: python make_cnf.py NAME K DOMAIN [--sym] OUT.cnf      NAME as in inequalities.py, DOMAIN one of cube, U, realizable, star
       python make_cnf.py NAME K DOMAIN [--sym] --solve [MAP.tsv]      decide with CaDiCaL through pysat (no proof) and, if satisfiable, write the map as a certificate for check_map.py"""
import json
import os
import sys

import numpy as np
from pysat.card import CardEnc, EncType

import inequalities as I
from core import irreducible_partners, occurrence_words, realizable_domain, sides, stabilizer, star_domain, term_name, u_domain, write_map


def build(name, k, domain, sym, emit):
    q, N = I.ALL[name]
    lhs, rhs = sides(q, N)
    L, R = len(lhs), len(rhs)
    alpha = np.array([k * c for _, c in lhs], dtype=np.int64)
    beta = np.array([k * c for _, c in rhs], dtype=np.int64)
    Rt = int(beta.sum())
    A = {"cube": lambda: np.arange(1 << L, dtype=np.int64), "U": lambda: u_domain(lhs, N), "realizable": lambda: realizable_domain(lhs), "star": lambda: star_domain(lhs, N)}[domain]()
    idx = -np.ones(1 << L, dtype=np.int64)
    idx[A] = np.arange(A.size)
    words = occurrence_words(lhs, N) + [0]
    assert all(idx[w] >= 0 for w in words)

    group = [p for p in stabilizer(q, N) if p[N] == N] if sym else [tuple(range(N + 1))]
    lm, rm = [m for m, _ in lhs], [m for m, _ in rhs]
    G = []
    for p in group:
        img = lambda m: sum(1 << p[i] for i in range(N) if m >> i & 1)
        G.append((np.array([lm.index(img(m)) for m in lm]), np.array([rm.index(img(m)) for m in rm])))

    def act(lp, x):
        out = np.zeros_like(x)
        for l in range(L):
            out |= ((x >> l) & 1) << lp[l]
        return out
    GA = [idx[act(lp, A)] for lp, _ in G]
    assert all((g >= 0).all() for g in GA), "the group does not preserve the domain"

    # (point, term) orbits by union-find
    par = np.arange(A.size * R)

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for (lp, cp), ga in zip(G, GA):
        src = (np.arange(A.size)[:, None] * R + np.arange(R)[None, :]).ravel()
        dst = (ga[:, None] * R + cp[None, :]).ravel()
        for a, b in zip(src.tolist(), dst.tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                par[max(ra, rb)] = min(ra, rb)
    root = np.array([find(a) for a in range(A.size * R)])
    reps, inv = np.unique(root, return_inverse=True)
    rc = reps % R
    assert all((beta[np.flatnonzero(inv == o) % R] == beta[rc[o]]).all() for o in range(reps.size)), "orbit mixes terms with different coefficients"
    off = np.zeros(reps.size, dtype=np.int64)
    off[1:] = np.cumsum(beta[rc])[:-1]
    nv = 1 + int(beta[rc].sum())  # variable 1 is TRUE

    def lit(u, r, j):  # [n_r(u) >= j + 1], j = 0, ..., beta_r - 1
        return 2 + int(off[inv[u * R + r]]) + j

    emit([1])
    for o in range(reps.size):
        r = int(rc[o])
        for j in range(int(beta[r]) - 1):
            emit([-(2 + int(off[o]) + j + 1), 2 + int(off[o]) + j])
    pinval = {}
    for i, w in enumerate(words):
        u = int(idx[w])
        for r in range(R):
            t = int(beta[r]) if (i < N and rhs[r][0] >> i & 1) else 0
            for j in range(int(beta[r])):
                v, s = lit(u, r, j), (1 if j < t else -1)
                assert pinval.get(v, s) == s, "pins conflict with the identification"
                if v not in pinval:
                    pinval[v] = s
                    emit([s * v])

    pt_rep = np.min(np.stack(GA), axis=0)
    xreps = np.flatnonzero(pt_rep == np.arange(A.size))
    pairs = {}
    n_irr = 0
    for xi in xreps.tolist():
        ys = irreducible_partners(A, xi, L)
        n_irr += len({int(g[xi]) for g in GA}) * ys.size
        for yi in ys.tolist():
            pairs.setdefault(min(tuple(sorted((int(g[xi]), int(g[yi])))) for g in GA), None)
    encoded = dropped = 0
    for ui, vi in pairs:
        d = int(alpha[np.flatnonzero(((A[ui] ^ A[vi]) >> np.arange(L)) & 1)].sum())
        if d >= Rt:
            dropped += 1
            continue
        D = []
        for r in range(R):
            for j in range(int(beta[r])):
                a, b = lit(ui, r, j), lit(vi, r, j)
                if a == b:
                    continue
                nv += 1
                D.append(nv)
                emit([-a, b, nv])
                emit([a, -b, nv])
        if len(D) <= d:
            continue
        enc = CardEnc.atmost(lits=D, bound=d, top_id=nv, encoding=EncType.cardnetwrk)
        nv = max(nv, enc.nv)
        for c in enc.clauses:
            emit(c)
        encoded += 1
    stats = {"inequality": name, "k": k, "domain": domain, "points": int(A.size), "L": L, "R": R, "k|beta|": Rt, "group": len(G), "point orbits": int(xreps.size),
             "irreducible pairs": n_irr // 2, "pair classes": len(pairs), "pairs dropped (d >= k|beta|)": dropped, "pairs with a cardinality constraint": encoded, "variables": nv}
    decode = lambda model: (A, [[sum(model[lit(u, r, j) - 1] > 0 for j in range(int(beta[r]))) for r in range(R)] for u in range(A.size)], lhs, rhs, N)
    return stats, decode


def write_cnf(name, k, domain, sym, out):
    body = out + ".body"
    ncl = 0
    with open(body, "w") as fh:
        def emit(c):
            nonlocal ncl
            fh.write(" ".join(map(str, c)) + " 0\n")
            ncl += 1
        stats, _ = build(name, k, domain, sym, emit)
    with open(out, "w") as fo, open(body) as fh:
        fo.write(f"p cnf {stats['variables']} {ncl}\n")
        for chunk in iter(lambda: fh.read(1 << 24), ""):
            fo.write(chunk)
    os.remove(body)
    stats["clauses"] = ncl
    return stats


def solve(name, k, domain, sym, mapfile):
    from pysat.solvers import Cadical153
    clauses = []
    stats, decode = build(name, k, domain, sym, clauses.append)
    stats["clauses"] = len(clauses)
    with Cadical153(bootstrap_with=clauses) as s:
        sat = s.solve()
        model = s.get_model() if sat else None
    stats["verdict"] = "SATISFIABLE" if sat else "UNSATISFIABLE"
    if sat and mapfile:
        A, vals, lhs, rhs, N = decode(model)
        cert = {"inequality": name, "k": k, "domain": "cube" if domain == "cube" else domain, "lhs": [term_name(m, N) for m, _ in lhs], "rhs": [term_name(m, N) for m, _ in rhs],
                "points": [int(x) for x in A], "values": vals}
        write_map(mapfile, cert)
    return stats


def main(args):
    sym = "--sym" in args
    args = [a for a in args if a != "--sym"]
    name, k, domain = args[0], int(args[1]), args[2]
    if args[3] == "--solve":
        stats = solve(name, k, domain, sym, args[4] if len(args) > 4 else None)
    else:
        stats = write_cnf(name, k, domain, sym, args[3])
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
