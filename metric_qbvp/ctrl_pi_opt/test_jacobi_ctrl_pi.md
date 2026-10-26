# `test_jacobi_ctrl_pi.py`: one encoding, all three frames by controlled-π

## 1. What it computes

For every interior edge of each frame $f \in \{z, y, x\}$:

$$
a^{\text{new}}_f(p) = \beta\,\frac{S_f(p) + J_f(p)}{\Delta_f(p)} + (1-\beta)\,a_f(p),
\qquad
S_f(p) = \sum_{\tau=1}^{12} s_\tau\,F_\tau\,\text{field}_\tau(p+\delta_\tau).
$$

Unlike `test_jacobi_all_frames_weighted.py`, the data is encoded **once**, in the common storage layout $(z, x, y)$. Each frame's view of it is created **inside the circuit** by controlled-π rotations of the index qubits.

---

## 2. Why the blocks must be edge–face products

A component plays a different role in each frame, and so it is multiplied by different faces:

| | z-frame | y-frame | x-frame |
|---|---|---|---|
| $A_x$ is | $b$, times $F_y$ | $c$, times $F_z$ | $a$, times $F_z$ and $F_y$ |
| $A_y$ is | $c$, times $F_x$ | $a$, times $F_x$ and $F_z$ | $b$, times $F_z$ |
| $A_z$ is | $a$, times $F_y$ and $F_x$ | $b$, times $F_x$ | $c$, times $F_y$ |

A single block "coef·$A_x$" can't serve all three frames, and a π rotation only relabels axes; it can't change which face multiplies a field. What **is** frame-independent is the set of **12 edge–face products**: each component times each of its 4 incident faces. Each frame needs 8 of the 12, and each product is used by exactly two frames.

The code doesn't assume this; it **checks** it. Every frame's 8 product blocks are mapped back to the storage layout and deduplicated, and the code asserts that exactly 12 distinct blocks remain (`problem()`, step 1).

$1/\Delta$ can't be folded into these shared blocks: the same product is divided by the $\Delta$ of a z-edge in one frame and of an x-edge in another. So $1/\Delta$ is applied **after the shift**, at the target edge, by a gate (W1).

---

## 3. Registers (q = 2, so a 4³ grid, 24 qubits)

| Register | Qubits | Role |
|---|---|---|
| `gk, gi, gj` | 6 | edge address; storage layout before ctrl-π, frame layout after it |
| `term` | 4 | star-term code that drives the shifter (computed by MAP, cleared by UNMAP) |
| `flag`, `carry` | 3 + 1 | star-shifter work qubits |
| `L` | 5 | **block label**: 0–11 products, 12–14 $J_z, J_y, J_x$, 15–17 $a_d\Delta_d$ (18–31 empty) |
| `frame` | 2 | $\lvert0\rangle$ = z, $\lvert1\rangle$ = y, $\lvert2\rangle$ = x ($\lvert3\rangle$ empty) |
| `copy` | 1 | a product used by **two** terms in one frame (e.g. $Q_b^{\Phi0}$ by E_bc_k and W_bc_k) gets two branches |
| `w1`, `w2` | 2 | weight ancillas: $1/\Delta$ and the constants |

q = 2 keeps the statevector at $2^{24}$. At q = 3 the same circuit needs 27 qubits (about 2 GB) and is slow to simulate. The material is **random in every cell**, which makes every face weight different and so tests the frame dependence hard. On a 3×3×3 cell grid the L-shape would cover only one cell.

---

## 4. Encoding (`problem`)

$$
\lvert\Psi_0\rangle = \frac{1}{\mathcal N}\sum_{L} \lvert L\rangle \otimes \lvert B_L\rangle ,
$$

with all blocks $B_L$ in the storage layout:

| L | $B_L$ |
|---|---|
| 0–11 | the 12 products $A_d\cdot F_c$, without $1/\Delta$ |
| 12, 13, 14 | $J_z$, $J_y$, $J_x$ |
| 15, 16, 17 | $a_d\cdot\Delta_d$ for $d = z, y, x$ |

**The $a\cdot\Delta$ trick:** W1 multiplies **every** branch by $1/\Delta_f(p)$. Pre-multiplying the retained term by $\Delta$ makes that factor cancel, leaving $(1-\beta)\,a$. The term $a_d$ is only used by its own frame, at its own edge, so this is exact.

---

## 5. The circuit (`build_body`)

| Stage | Gates | Effect |
|---|---|---|
| **FRAME** | $R_Y(2\arccos\sqrt{2/3})$, X·CH·X | $\tfrac{1}{\sqrt3}(\lvert0\rangle+\lvert1\rangle+\lvert2\rangle)$ on `frame`: the same data in three frame branches |
| **COPY** | H | two copies of every branch, for doubly-used products |
| **ctrl-π** | 12 controlled-SWAPs | in branch y: $(z,x,y)\to(y,z,x)$; in branch x: $(z,x,y)\to(x,y,z)$; branch z unchanged |
| **MAP** | MCX lookup on (L, frame, copy) | writes the term code of the role that block plays in that frame (constant size) |
| **STAR SHIFTER†** | ripple-carry star shifter, inverted | every branch: value at $s$ → $p = s-\delta_\tau$, all frames at once |
| **UNMAP** | same lookup | term register back to $\lvert0000\rangle$ |
| **W1** | uniformly controlled $R_Y$ on `w1`, controls (index, frame) | amplitude $\frac{1/\Delta_f(p)}{\max(1/\Delta)}$ on $\lvert w_1=0\rangle$, at the **target** edge |
| **W2** | uniformly controlled $R_Y$ on `w2`, controls (L, frame, copy) | amplitude $s_\tau\beta$ (star), $+\beta$ ($J_f$), $1-\beta$ ($a_f\Delta_f$), $0$ (blocks not used in that frame) |
| **SUM** | H on `L` and `copy` (**not** on `frame`) | adds all branches of a frame onto $\lvert L=0,\text{copy}=0\rangle$, factor $1/8$ |

---

## 6. Readout

$$
\langle f,\ \text{rest}=0,\ p\,\vert\,\Psi\rangle
= \frac{a^{\text{new}}_f(p)}{8\sqrt{6}\;\mathcal N\;\max(1/\Delta)} ,
$$

where $1/\sqrt3$ comes from FRAME, $1/\sqrt2$ from COPY, $1/8$ from SUM, $1/\mathcal N$ from the encoding and $\max(1/\Delta)$ from W1's normalisation. The result is in the frame's layout. `to_storage()` converts it to $a_x, a_y, a_z$ in their native storage shapes.

---

## 7. Results (one Jacobi step, random fields, random $J$, random $\nu$)

| | max $\lvert a^{\text{new}}_{\text{quantum}} - a^{\text{new}}_{\text{classical}}\rvert$ |
|---|---|
| z-frame | $2.4\cdot10^{-15}$ |
| y-frame | $3.8\cdot10^{-15}$ |
| x-frame | $1.6\cdot10^{-15}$ |

The run takes one circuit for all three frames, 7.9 s, with success probability $1.7\cdot10^{-4}$.

---

## 8. Files for checking

**Results**, in `results/jacobi_ctrl_pi/`:

| File | Array | Shape (storage layout $z, x, y$) |
|---|---|---|
| `ax_quantum.txt`, `ax_classical.txt` | $a_x$ after the step | $(n,\ n-1,\ n) = (4, 3, 4)$ |
| `ay_quantum.txt`, `ay_classical.txt` | $a_y$ | $(n,\ n,\ n-1) = (4, 4, 3)$ |
| `az_quantum.txt`, `az_classical.txt` | $a_z$ | $(n-1,\ n,\ n) = (3, 4, 4)$ |

Each file keeps the 3-D shape: one block per $k$ (the first index, z), with rows = x and columns = y. Interior edges hold the updated values, and boundary edges are 0, as in the solver. To read one back:

```python
import numpy as np
def load_field(path, shape):
    rows = np.loadtxt(path)                 # comment lines (#) are skipped
    return rows.reshape(shape)
az_q = load_field("results/jacobi_ctrl_pi/az_quantum.txt", (3, 4, 4))
```

**Circuits**, in `circuits/jacobi_ctrl_pi/`: `jacobi_ctrl_pi_blocks.svg/.qpy` (stages as boxes) and `jacobi_ctrl_pi_gates.svg/.qpy` (gate level).

---

## 9. What is idealised, and how it scales

- **Encoding** uses `set_statevector` (an Aer instruction); a hardware version would use `StatePreparation`. The products $A_d\cdot F_c$ are formed classically; coherently they would come from the field state through a Hodge oracle.
- **W1 and W2 are lookup tables** (`UCRYGate`).
  - W2 is constant-size: $2^8$ angles, independent of the grid.
  - W1 has $4\cdot n^3$ angles, i.e. it grows with the grid. For piecewise-constant materials, $\Delta$ takes few values, so W1 can be replaced by comparators on the index plus one controlled rotation per value.
- **Success probability** is low, about $10^{-4}$: the frame/copy/label overheads and the normalisation of $1/\Delta$. Amplitude amplification would raise it.
- **Readout** uses exact amplitudes from the statevector; on hardware this needs amplitude estimation.

---

## 10. Code map

| Function | Purpose |
|---|---|
| `problem` | finds the 12 products (with the uniqueness check), builds the encoded vector, the MAP table and the W1/W2 angles |
| `frame_prep`, `ctrl_pi`, `lookup`, `build_body` | the circuit stages |
| `jacobi_step_quantum` | one circuit run → $a^{\text{new}}$ for z, y, x |
| `jacobi_step_classical` | reference $\beta(S+J)/\Delta + (1-\beta)a$ (the paper's 3-D update, source with $+$) |
| `to_storage`, `save_field_txt` | convert to $a_x, a_y, a_z$ and write them shape-preserving |
| `named_circuits` | block- and gate-level circuits for `circuit_saving.save_circuits` |

Shared pieces: `star_shifter.py` (shifter, `TERMS`, offsets), `test_combined_frames_aer.py` (`blocks`, `TERM_BLOCK`), `circuit_saving.py`.

**Run:** `metric/bin/python test_jacobi_ctrl_pi.py` (about 15 s).
