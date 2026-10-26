# `test_jacobi_all_frames_weighted.py`: one Jacobi step for all three frames in one circuit

## 1. What it computes

For every interior edge $p$ of each frame ($z$, $y$, $x$), the damped Jacobi update

$$
a^{\text{new}}(p) \;=\; \beta\,\frac{S(p) + J(p)}{\Delta(p)} \;+\; (1-\beta)\,a(p),
\qquad
S(p) = \sum_{\tau=1}^{12} s_\tau\, F_\tau\, \text{field}_\tau(p+\delta_\tau),
$$

This is the 3-D update of the paper (`paper_section/product_blocks.tex`, Eq. `jacobi`), shown here for the z-frame:

$$
A_z^{s+1} = \frac{\beta}{\Delta}\big(E+W+N+S+J_z\big) + (1-\beta)\,A_z^{s}.
$$

The source enters with a **plus** sign. Since $\Delta a - S = M a$ is the discrete curl-curl operator, the fixed point satisfies $M a = J$. An earlier version of this test used $S - J$. That solves $M a = -J$, which gives the same field with the opposite sign. The circuit and the classical reference now both use $+J$, and the old $-J$ sign gate has been removed (§5).

where
- $S$ is the star sum: the 12 face-weighted neighbour terms, with signs $s_\tau = \pm1$ and offsets $\delta_\tau$;
- $\Delta(p)$ is the sum of the four face reluctivities around the edge;
- $J$ is the source;
- $a$ is the old value of the component being updated.

**One circuit run gives $a^{\text{new}}$ at every edge of all three frames.** The result agrees with the classical update to machine precision.

---

## 2. Idea in one line

Encode the three kinds of term, $\text{coeff}_{nf}\cdot a/\Delta$, $J/\Delta$ and $a$, in separate **subspaces**. Then shift them, give each subspace its constant ($\beta$ for the star terms and $J$, $1-\beta$ for $a$), and add everything with Hadamards.

---

## 3. Registers (q = 3, so an 8³ grid)

| Register | Qubits | Role |
|---|---|---|
| `gk, gi, gj` | 9 | edge address $(k, i, j)$ in the frame's layout |
| `term` | 4 | which of the 12 star terms $(\alpha, \sigma, r_1, r_0)$; `0000` for $J$ and $a$ |
| `flag`, `carry` | 3 + 2 | work qubits of the star shifter (start and end in $\lvert0\rangle$) |
| `type` | 2 | **subspace**: $\lvert0\rangle$ = star terms, $\lvert1\rangle$ = $J/\Delta$, $\lvert2\rangle$ = $a$; basis state $\lvert3\rangle$ is empty (amplitude 0) |
| `w` | 1 | weight ancilla for the constants $\beta$, $1-\beta$ |
| `frame` | 2 | which component: $\lvert0\rangle$ = $z$, $\lvert1\rangle$ = $y$, $\lvert2\rangle$ = $x$; basis state $\lvert3\rangle$ is empty (amplitude 0) |

| **total** | **23** | |

Both qubits of `type` and both qubits of `frame` are used: 3 values need 2 qubits, which leaves one of the 4 basis states empty. No gate acts on `frame`. It only **labels** the components and keeps their results apart (see §5).

Flat statevector index:
$\;p + 2^{9}\,\text{term} + 2^{18}\,\text{type} + 2^{21}\,\text{frame}$, with the flags, carries and `w` equal to 0.

### 3.1 The term codes

The 4 term qubits are $(t_0, t_1, t_2, t_3) = (\alpha, \sigma, r_1, r_0)$, so the code is

$$
\text{term} = \alpha + 2\sigma + 4r_1 + 8r_0 .
$$

| bit | name | meaning |
|---|---|---|
| $t_0$ | $\alpha$ | axis of the shift: 0 = E/W (local $i$-axis), 1 = N/S (local $j$-axis) |
| $t_1$ | $\sigma$ | side: 0 = E or N (the $+$ side), 1 = W or S (the $-$ side) |
| $t_2, t_3$ | $r_1, r_0$ | member of the face term: $a$ = $(r_1, r_0) = (0, 1)$, $b/c$ at $k$ = $(0, 0)$, $b/c$ at $k{+}1$ = $(1, 0)$ |

The code is chosen so that the star shifter can read the shift off the bits almost directly:
- $\delta k = r_1$;
- the in-plane shift is active when $\sigma \lor r_0$;
- its axis is $\alpha$;
- its direction is $-$ when $\sigma = 1$.

All 12 star terms, for any frame $(k,i,j)$, with $(a, b, c)$ and $(\Phi, \Psi)$ as in `test_combined_frames_aer.py`:

| term | $\alpha$ | $\sigma$ | $r_1$ | $r_0$ | code | block $Q_\tau$ | sign $s_\tau$ | offset $\delta_\tau=(\delta k,\delta i,\delta j)$ | source at $p+\delta_\tau$ |
|---|---|---|---|---|---|---|---|---|---|
| E_bc_k  | 0 | 0 | 0 | 0 | **0**  | `Qb_y0` = $\Phi\, b$       | $+$ | $(0,0,0)$   | $[k,i,j]$ |
| N_bc_k  | 1 | 0 | 0 | 0 | **1**  | `Qc_x0` = $\Psi\, c$       | $+$ | $(0,0,0)$   | $[k,i,j]$ |
| W_bc_k  | 0 | 1 | 0 | 0 | **2**  | `Qb_y0` = $\Phi\, b$       | $-$ | $(0,-1,0)$  | $[k,i{-}1,j]$ |
| S_bc_k  | 1 | 1 | 0 | 0 | **3**  | `Qc_x0` = $\Psi\, c$       | $-$ | $(0,0,-1)$  | $[k,i,j{-}1]$ |
| E_bc_k1 | 0 | 0 | 1 | 0 | **4**  | `Qb_ym` = $\Phi^{-}_k\, b$ | $-$ | $(+1,0,0)$  | $[k{+}1,i,j]$ |
| N_bc_k1 | 1 | 0 | 1 | 0 | **5**  | `Qc_xm` = $\Psi^{-}_k\, c$ | $-$ | $(+1,0,0)$  | $[k{+}1,i,j]$ |
| W_bc_k1 | 0 | 1 | 1 | 0 | **6**  | `Qb_ym` = $\Phi^{-}_k\, b$ | $+$ | $(+1,-1,0)$ | $[k{+}1,i{-}1,j]$ |
| S_bc_k1 | 1 | 1 | 1 | 0 | **7**  | `Qc_xm` = $\Psi^{-}_k\, c$ | $+$ | $(+1,0,-1)$ | $[k{+}1,i,j{-}1]$ |
| E_a     | 0 | 0 | 0 | 1 | **8**  | `Qa_ym` = $\Phi^{-}_i\, a$ | $+$ | $(0,+1,0)$  | $[k,i{+}1,j]$ |
| N_a     | 1 | 0 | 0 | 1 | **9**  | `Qa_xm` = $\Psi^{-}_j\, a$ | $+$ | $(0,0,+1)$  | $[k,i,j{+}1]$ |
| W_a     | 0 | 1 | 0 | 1 | **10** | `Qa_y0` = $\Phi\, a$       | $+$ | $(0,-1,0)$  | $[k,i{-}1,j]$ |
| S_a     | 1 | 1 | 0 | 1 | **11** | `Qa_x0` = $\Psi\, a$       | $+$ | $(0,0,-1)$  | $[k,i,j{-}1]$ |

Notation in the block column:
- $\Phi^{-}_i\,a$ means $\Phi[k,i{-}1,j]\,a[k,i,j]$ (the `…m` blocks hold the face on the minus side along that axis);
- $\Phi^{-}_k$ and $\Psi^{-}_k$ shift the face by $-1$ along $k$; $\Psi^{-}_j$ shifts it along $j$.

After the shift, each block value sits at $p$ with the correct face: for example code 8 gives $\Phi[k,i,j]\,a[k,i{+}1,j]$, the first term of $E$.

Codes 12–15 ($r_1 = r_0 = 1$) are not used and have amplitude 0. The $J$ and $a$ subspaces (types 1 and 2) use term code `0000`. That is a no-shift code, because $\delta = (0,0,0)$ for $r_1 = r_0 = 0$ and $\sigma = 0$. They share the code with E_bc_k, but they live in a different `type` subspace, so they don't mix with it until the final sum.

The signs $s_\tau$ come from the gate SIGN star (§5): $-1$ on codes 2, 3, 4, 5.

---

## 4. The encoding (`encode`)

For each frame $f$, with its fields $(a, b, c)$ and face arrays $(\Phi, \Psi)$ taken from `test_combined_frames_aer.py`, all in that frame's own layout:

| Subspace | Amplitude block | Term code |
|---|---|---|
| type 0 | $Q_\tau[s]\,/\,\Delta(s-\delta_\tau)$ for each of the 12 star terms | the term's code |
| type 1 | $J[s]\,/\,\Delta(s)$ | `0000` |
| type 2 | $a[s]$ | `0000` |

Here $Q_\tau = F_\tau \cdot \text{field}_\tau$ are the edge-aligned product blocks, i.e. $\text{coeff}_{nf}\cdot a$.

**Why $\Delta(s-\delta_\tau)$?** The value at source edge $s$ will be moved by the shifter to the target edge $p = s - \delta_\tau$, and the division by $\Delta$ belongs to the **target** edge. So each block is divided by $\Delta$ evaluated at its value's destination:

```python
blk = pad(Q[...]) * np.roll(invD, expected_offset(...), axis=(0, 1, 2))   # invD shifted to the target
```

All amplitudes are normalised together by one factor $\mathcal N$, and the state is loaded with `qc.set_statevector(psi0)` (see §8).

---

## 5. The circuit (`build_body`)

```
SIGN star  →  STAR SHIFTER†  →  β, 1−β  →  SUM H⊗6
```

| Stage | Gates | Effect |
|---|---|---|
| **SIGN star** | X·CZ·X, X·CCZ·X on `term` | $-1$ on the four negative star terms: W_bc_k (2), S_bc_k (3), E_bc_k1 (4), N_bc_k1 (5) |
| **STAR SHIFTER†** | ripple-carry star shifter, inverted | each star branch: value at $s$ → $p = s-\delta_\tau$. $J$ and $a$ (code `0000`) don't move. All three frames at once, since frame-local offsets are identical |
| **β, 1−β** | `RY(2·acos β)` on `w`, controlled by `type₁ = 0`; `RY(2·acos(1−β))` on `w`, controlled by `type₁ = 1` | amplitude on $\lvert w=0\rangle$: $\beta$ for types 0, 1 (star terms and $+J/\Delta$); $1-\beta$ for type 2 |

$J$ needs no gate of its own. It enters with $+\beta$, exactly like the star terms, so the β gate for `type₁ = 0` covers both.
| **SUM** | H on the 4 `term` + 2 `type` qubits | adds all branches onto $\lvert\text{term}=0,\text{type}=0\rangle$ with factor $1/8$ |

Why the constants work: $R_Y(2\arccos c)\lvert0\rangle = c\lvert0\rangle + \sqrt{1-c^2}\,\lvert1\rangle$. Keeping only $\lvert w = 0\rangle$ multiplies a branch by $c$.

The frame register is never acted on. It only **labels** the three components, so each frame's result ends up in its own block of the statevector:

- the shifter does not need it: each frame is encoded in its own layout, so the 12 offsets are the same in every frame;
- the constants do not need it: $\beta$ and $1-\beta$ are the same for every frame;
- the summation **leaves it out on purpose**: H acts only on `term` and `type`, so terms are added within a frame, never across frames.

Without the frame register, the branches of all three frames would share one subspace and be added together, mixing $A_z$, $A_y$ and $A_x$. With it, one run gives three separate updates. No controlled-SWAP rotations are needed, because each frame is already encoded in its own layout.

---

## 6. Readout

After the circuit, the amplitude at $\lvert\text{frame}=f,\ w=0,\ \text{type}=0,\ \text{term}=0,\ \text{ancillas}=0,\ p\rangle$ is

$$
\frac{1}{8\,\mathcal N}\Big[\beta\sum_\tau s_\tau\frac{Q_\tau(p+\delta_\tau)}{\Delta(p)} \;+\; \beta\,\frac{J(p)}{\Delta(p)} \;+\; (1-\beta)\,a(p)\Big]
\;=\; \frac{a^{\text{new}}_f(p)}{8\,\mathcal N}.
$$

In code (`jacobi_step_all_frames`):

```python
ro = sv[fi*2**FR[0] : fi*2**FR[0] + n**3]          # frame f, everything else 0
a_new[f] = (from_flat(ro).real * 8 * norm)[K, I, J_]
```

`[K, I, J_]` keeps the interior edges: $k = 0..n-2$ and $i, j = 1..n-2$.

---

## 7. Results

| Check | Result |
|---|---|
| One step, random fields and random $J$ (with $+J$), **one run for all 3 frames** | max error $2.7\cdot10^{-15}$ (z), $1.8\cdot10^{-15}$ (y), $3.6\cdot10^{-15}$ (x); 0.8 s |
| Success probability (all frames together) | $P = 0.0054$ |
| Full L-shape solve (J = 0, the notebook's boundary conditions) | converges at **step 85**, like the classical solver; $\max\lvert A_z^{q} - A_z^{c}\rvert = 3.0\cdot10^{-13}$; 86 circuit runs, 54 s |

---

## 8. What is idealised

- **Encoding:** the products $\text{coeff}_{nf}\cdot a/\Delta$ are computed classically and loaded with `set_statevector`. That is an Aer simulator instruction, not a gate. A hardware version would use `StatePreparation` (real gates; tested elsewhere, error ~$10^{-8}$) or produce the blocks coherently with a Hodge oracle.
- **$1/\Delta$ is in the encoding**, not in a gate. The constants $\beta$ and $1-\beta$ are applied in the circuit; a position-dependent $1/\Delta(p)$ would need a weight oracle (e.g. comparators for piecewise-constant materials).
- **Readout:** exact amplitudes from the statevector. On hardware this needs amplitude estimation.
- **Boundary conditions** are applied classically after each step (`apply_bc_az`).

---

## 9. Code map

| Function | Purpose |
|---|---|
| `sign_star` | the $-1$ signs of the four negative star terms (codes 2–5) |
| `constants` | the two controlled $R_Y$ gates: $\beta$ (star terms and $J$), $1-\beta$ ($a$) on `w` |
| `build_body` | SIGN → STAR SHIFTER† → constants → H⊗6 |
| `encode` | amplitude blocks $[\text{frame}, \text{type}, \text{term}, \text{index}]$ and $\mathcal N$ |
| `jacobi_step_all_frames` | one circuit run → $a^{\text{new}}$ for z, y, x |
| `classical` | reference: $\beta(S+J)/\Delta + (1-\beta)a$ |
| `solve`, `step_quantum`, `step_classical` | full L-shape solve with each step |
| `named_circuits` | block- and gate-level circuits with named registers |

Shared pieces come from `test_combined_frames_aer.py` (fields, faces, blocks, classical star sum), `star_shifter.py` (`build_proposed`, `TERMS`, `expected_offset`) and `circuit_saving.py` (`save_circuits`).

**Run:** `metric/bin/python test_jacobi_all_frames_weighted.py` (about 1 minute).

**Saved circuits** (`circuits/jacobi_all_frames_weighted/`): `jacobi_all_frames_blocks.svg/.qpy` (overview) and `jacobi_all_frames_gates.svg/.qpy` (gate level).
