# Holographic entropy inequalities without unscaled contraction proofs: code

Code and certificates for the computations in the paper *Holographic entropy inequalities without unscaled contraction proofs*. Each script reproduces the numbers of the statements listed next to it; numbers refer to the paper.

| Script | Reproduces |
|---|---|
| `inequalities.py` | The inequalities of the paper, from its displays and the identities of its text, and their invariants: the Bell values of $B_7^*$ (Section 3), the data of $U_6$ and of the point $z$ in the proof of Proposition 4.3(iii), including that no integer value at $z$ satisfies (4.5) (Section 4.1), the Bell values of $Q_8^A$ (Section 4.2), the data of $Q_8$ and the deletion giving (4.8) (Section 4.3), the stabilizer of $E_0$ (Section 4.4), Table 1 (with $\lvert\mathcal U\rvert$, superbalance and the antichain condition of Theorem 4.11), the data of $F_7$ in Section 5.1, Table 2, the stabilizers in Section 5.3 and the half-sum of Remark 5.9. The full coefficient lists (Appendix C) are in `certificates/inequalities.txt`, which the script checks against the definitions |
| `check_identities.py` | Proposition 3.2(i) (43 instances), Remark 3.5 (46 instances), Proposition 4.3(i), $42R_8$ as a sum of 147 instances of MMI (Section 4.3), Proposition 4.12, identities (5.5) and (5.6), and the identities (5.10) and $2E_s=V_7'+P_7+2M_s$ of Remark 5.9 |
| `check_map.py` | The maps of Tables 4 and 5: Proposition 3.2(ii) (all 32,640 pairs), Remark 3.5, Section 4.2, Proposition 4.3(ii) (all 167,910 pairs), the validity maps of Theorem 4.11, Theorem 5.2 (all 61,001,535 pairs), Proposition 5.4, Theorem 5.7(i), Remarks 5.8 and 5.9, Proposition 6.1(ii) (all edges of the cubes, and all 4,250,070 pairs on $\mathcal A(E_4)$), and the maps at $k_{\min}$ in Table 3 (Appendix A.2) |
| `check_graphs.py` | Theorem 5.2 (two lists of 126 tight graph models, rank 126 modulo three primes), Remark 5.8 (125 tight graph models each for $V_7$ and $V_7'$), and the upper bounds of Table 1 and Proposition 4.12 (Appendix A.3) |
| `make_cnf.py` | The SAT encodings of Appendices A.1 and A.4, for every statement that a map does not exist (below) |
| `stars.py` | Table 3 (the single point of each star polytope and the identities of Lemma 6.8, Theorem 6.14(b) and Appendix C.2), conditions (W1) and (W2) of Theorem 6.14 for the lists of Appendix C.2, Theorem 6.17(a), (b) for $j=0,1,2$, Example 6.19, the polytope of $Q'$ (Section 6.2) and the Bell values of Section 6.5 |
| `compare_F7.py` | Remark 5.3, Appendix C.3 and Table 6 |

`core.py` holds the definitions the scripts share. The certificates are in `certificates/`: the maps (`maps/`, one tab-separated table per map, named by scaling, inequality, domain and type: a `#` header names the inequality, the scaling, the domain and the LHS and RHS terms, and each row gives a point as its 0/1 word over the LHS terms, in the listed order, followed by the value of the map on each RHS term, an integer or a fraction), the graph models (`graphs/`), the instances of subadditivity and MMI of the identities (`identities.json`) and the coefficient lists (`inequalities.txt`).

The paper and this code were produced by large language models; the paper's introduction states how, and what has and has not been checked.

## Running

Python 3.12 with the versions in `requirements.txt`:

```
pip install -r requirements.txt
python inequalities.py
python check_identities.py
python check_map.py
python check_graphs.py
python stars.py
python compare_F7.py
```

Each script prints what it checks and stops with an error if a check fails.

## Non-existence of maps

The DRAT proofs are not included, because they can take several gigabytes. `make_cnf.py` writes the formula for each statement, a proof-producing SAT solver such as [CaDiCaL](https://github.com/arminbiere/cadical) writes the proof, and [drat-trim](https://github.com/marijnheule/drat-trim) checks it:

```
python make_cnf.py E0 1 U e0.cnf
cadical e0.cnf e0.drat
drat-trim e0.cnf e0.drat
```

The solver must report `s UNSATISFIABLE` and drat-trim `s VERIFIED`. The arguments are the inequality (named as in `inequalities.py`), the scaling $k$, the domain (`cube`, `U`, `realizable`, or `star` for the $N+2$ points $0$, $z$ and the occurrence words of Appendix A.1) and, for equivariant maps, `--sym`:

| Statement | Arguments |
|---|---|
| Proposition 3.2(iii), on the full cube | `B7* 1 cube` |
| Remark 3.5 | `B7 1 cube` |
| Section 4.2 | `Q8A 1 cube` |
| Proposition 4.3(iii) | `U6 1 realizable` |
| Theorem 4.11 | `Q8 1 U`, `Q8' 1 U`, `Q8'' 1 U`, `E0 1 U`, ..., `E5 1 U` |
| Theorem 5.7(ii) | `F7 1 realizable --sym` |
| Remark 5.8 | `V7 1 realizable --sym` |
| Remark 5.9 | `"(V7'+P7)/2" 1 U` |
| Proposition 6.1(i) | `Q8 3 U` |
| Table 3 | `Q' k cube` for $k=1,3$; `D0 k cube` for $k\le3$; `D1 k cube` for $k\le2$; `Z7 k cube` for $k\le7$; `G9 k star` for $k\le31$; `H9 k star` for $k\le17$; `H11 k star` for $k\le25$; `H15 k star` for $k\le35$; `H17 k star` for $k\le31$ |

`python make_cnf.py NAME K DOMAIN --solve` decides an instance without a proof, and `--solve MAP.tsv` writes the map of a satisfiable instance as a certificate for `check_map.py`; `certificates/maps/2Q8A_cube_integer.tsv` was obtained in this way.

## Licence

MIT, see `LICENSE`.
