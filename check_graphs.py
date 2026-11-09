"""Graph-model certificates (Appendix A.3): the tight graph models behind Theorem 5.2 and Remark 5.8, and the graph models that bound the steps of the splittings in Table 1 and Proposition 4.12.

A graph in certificates/graphs/ has vertices 0, ..., n-1, where vertex i < N is the boundary vertex of the i-th party, vertex N that of the purifier, and the others are bulk vertices, and integer edge weights. The entropy S(I) of a nonempty set I of parties is the minimum weight of a cut separating the boundary vertices of I from the other boundary vertices, computed as a maximum flow by two independent implementations (scipy and networkx), which must agree. The rank of a list of entropy vectors is computed modulo three primes; since reduction modulo p does not increase the rank of an integer matrix, rank r modulo p implies rank at least r over the rationals.

Usage: python check_graphs.py"""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

import inequalities as I
from core import add, subsets, vector

GRAPHS = Path(__file__).with_name("certificates") / "graphs"
PRIMES = (1000000007, 998244353, 2**61 - 1)


def entropy_scipy(n, edges, N):
    E = np.array(edges, dtype=np.int64)
    big = int(E[:, 2].sum()) + 1
    s, t = n, n + 1
    out = []
    for m in subsets(N):
        rows = list(E[:, 0]) + list(E[:, 1])
        cols = list(E[:, 1]) + list(E[:, 0])
        cap = list(E[:, 2]) * 2
        for v in range(N + 1):
            rows.append(s if (v < N and m >> v & 1) else v)
            cols.append(v if (v < N and m >> v & 1) else t)
            cap.append(big)
        M = csr_matrix((np.array(cap, dtype=np.int32), (rows, cols)), shape=(n + 2, n + 2))
        out.append(int(maximum_flow(M, s, t).flow_value))
    return out


def entropy_networkx(n, edges, N):
    big = sum(w for _, _, w in edges) + 1
    G = nx.DiGraph()
    G.add_nodes_from(range(n + 2))
    for u, v, w in edges:
        for a, b in ((u, v), (v, u)):
            G.add_edge(a, b, capacity=G[a][b]["capacity"] + w if G.has_edge(a, b) else w)
    s, t = n, n + 1
    out = []
    for m in subsets(N):
        H = G.copy()
        for v in range(N + 1):
            if v < N and m >> v & 1:
                H.add_edge(s, v, capacity=big)
            else:
                H.add_edge(v, t, capacity=big)
        out.append(nx.maximum_flow_value(H, s, t))
    return out


def entropies(path):
    data = json.loads(path.read_text())
    N = data["parties"]
    S = []
    for g in data["graphs"]:
        a, b = entropy_scipy(g["vertices"], g["edges"], N), entropy_networkx(g["vertices"], g["edges"], N)
        assert a == b, "max-flow implementations disagree"
        S.append(a)
    return data["graphs"], np.array(S, dtype=object)


def rank_mod(S, p):
    A = [[int(x) % p for x in row] for row in S]
    r = 0
    for c in range(len(A[0])):
        piv = next((i for i in range(r, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = pow(A[r][c], p - 2, p)
        A[r] = [x * inv % p for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [(x - f * y) % p for x, y in zip(A[i], A[r])]
        r += 1
    return r


def dot(q, N, S):
    v = vector(q, N)
    return [int(sum(a * b for a, b in zip(v, s))) for s in S]


def main():
    F7, V7, V7P = I.F7, I.V7, I.V7P
    for fname, name, q, size in (("F7_tight_1.json", "F7", F7, 126), ("F7_tight_2.json", "F7", F7, 126), ("V7_tight.json", "V7", V7, 125), ("V7p_tight.json", "V7'", V7P, 125)):
        graphs, S = entropies(GRAPHS / fname)
        assert len(graphs) == size and all(x == 0 for x in dot(q, 7, S)), fname
        ranks = [rank_mod(S, p) for p in PRIMES]
        assert ranks == [size] * 3, ranks
        extra = ""
        if name == "V7":
            assert all(x == 0 for x in dot(F7, 7, S))
            extra = "; F7 vanishes on all of them as well"
        print(f"{fname}: {size} graph models ({min(g['vertices'] for g in graphs)} to {max(g['vertices'] for g in graphs)} vertices), {name} . S = 0 for each, rank {size} modulo {', '.join(map(str, PRIMES))}{extra}", flush=True)

    # Steps of the splittings: a graph with next . S = 0 and X . S > 0, where Q = next + c X, shows that Q - t X is invalid for t > c.
    steps = (("Q8_step.json", "Q8 - t R8", I.Q8P, 8, I.R8, "2/3", "since 3(Q8 - t R8) = Q8' + (2 - 3t) R8"),
             ("E0_step.json", "E0 - t mu0", I.E[1], 7, I.MU[0], "2", "since E0 - t mu0 = E1 + (2 - t) mu0"),
             ("E1_step.json", "E1 - t mu1", I.E[2], 7, I.MU[1], "2", "since E1 - t mu1 = E2 + (2 - t) mu1"),
             ("E4_step.json", "E4 - t mu4", I.E[5], 7, I.MU[4], "2", "since E4 - t mu4 = E5 + (2 - t) mu4"))
    for fname, label, nxt, N, X, bound, why in steps:
        graphs, S = entropies(GRAPHS / fname)
        a, b = dot(nxt, N, S), dot(X, N, S)
        assert all(x == 0 for x in a) and all(x > 0 for x in b)
        nv = sorted({g["vertices"] for g in graphs})
        print(f"{fname}: {len(graphs)} graph model(s) with {nv[0]}{'' if len(nv) == 1 else ' to ' + str(nv[-1])} vertices (boundary vertices included), next inequality vanishes, subtracted vector takes the values {sorted(set(b))}: {label} is invalid for t > {bound} ({why})", flush=True)
    assert add((3, I.Q8), (-1, I.Q8P), (-2, I.R8)) == {}
    return 0


if __name__ == "__main__":
    sys.exit(main())
