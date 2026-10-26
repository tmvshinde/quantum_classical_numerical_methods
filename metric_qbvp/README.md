# metric_qbvp: quantum circuits for damped Jacobi relaxation of 3-D curl-curl magnetostatics

This folder holds quantum-circuit implementations of one **damped Jacobi step** for the edge-element (Whitney-1) discretisation of 3-D magnetostatics with a spatially varying material:

$$
\nabla\times\big(\nu\,\nabla\times\mathbf A\big) = \mathbf J, \qquad \nu = 1/\mu .
$$

The edge unknowns $a$ of each component ($A_x$, $A_y$, $A_z$) are updated with

$$
a^{\text{new}} = \beta\,\frac{S(a) + J}{\Delta} + (1-\beta)\,a ,
$$

where:
- $S$ is the sum of the 12 neighbouring "star" terms, each weighted by the reluctivity of the face it shares with the edge;
- $\Delta$ is the sum of the four face reluctivities around the edge.

The update is written per **frame** (z, y, x), and every star term is a **shift of an edge–face product array**. This is the formulation described in [`product_blocks.pdf`](product_blocks.pdf) and in L. Kettunen and T. M. Shinde, arXiv:2607.11410.

On the quantum side:
- **amplitudes** hold the products, $J$ and $a$;
- **one ripple-carry "star shifter"** applies all 12 neighbour shifts at once, in superposition;
- the constants $\beta$ and $1-\beta$ are applied by controlled $R_Y$ rotations on a weight ancilla;
- **Hadamards** add the terms together.

One circuit run returns the update for **all three components**.

## Folders

Each subfolder is self-contained: it has its own copies of the shared modules, so it can be run from inside that folder.

| Folder | Case | Main file | Qubits |
|---|---|---|---|
| [`all_frames_weighted/`](all_frames_weighted) | each frame encoded in its own layout; a frame register keeps the three components apart | `test_jacobi_all_frames_weighted.py` | 23 ($8^3$ grid) |
| [`ctrl_pi_opt/`](ctrl_pi_opt) | data encoded **once**; the frames are created in the circuit by controlled-π rotations; structured block labels keep the MAP/UNMAP small | `test_jacobi_ctrl_pi_opt.py` | 24 ($4^3$ grid) |
| [`L_shape_two_methods/`](L_shape_two_methods) | full solve of the L-shaped iron benchmark with both circuits, plus $\mathbf B=\nabla\times\mathbf A$ and plots | `L_shape_two_methods.ipynb` | 23 / 28 ($8^3$ grid) |

Shared modules, copied into each folder that needs them:

| File | Content |
|---|---|
| `star_shifter.py` | the 12-term star shifter: ripple-carry increments with the control as carry-in, ± by the complement trick ($T = 12q-4$, $4q+6$ qubits) |
| `test_combined_frames_aer.py` | the L-shape problem ($\mu_r = 10$, $8^3$): faces, product blocks, frame layouts, the classical star sum |
| `test_jacobi_ctrl_pi.py` | the original ctrl-π construction (encoding, W1, readout), which the optimised version builds on |
| `circuit_saving.py` | saves every circuit as `.svg` (+ `.qpy`) |

[`product_blocks.pdf`](product_blocks.pdf) is the write-up of the edge–face product-block formulation.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The simulations use dense statevectors. 23–24 qubits need about 0.3 GB, but the 28-qubit ctrl-π run in the notebook needs about **4 GB** per statevector (8 GB of RAM or more is recommended).

## Running

```bash
cd all_frames_weighted && python test_jacobi_all_frames_weighted.py     # ~1-2 min
cd ctrl_pi_opt        && python test_jacobi_ctrl_pi_opt.py              # ~10 s
cd L_shape_two_methods && jupyter notebook L_shape_two_methods.ipynb    # ~15 min (5 ctrl-pi steps at 28 qubits)
```

Outputs are written next to the scripts:
- circuit diagrams to `circuits/<test>/`;
- the fields $a_x, a_y, a_z$ to `results/<test>/`, as text files that keep the 3-D shape, one block per $k$.

Both folders are git-ignored.

## Results (Qiskit Aer statevector)

| Check | Result |
|---|---|
| all-frames: one step, random fields and $J$, all 3 frames in one run | max error vs. classical $\le 3.6\cdot10^{-15}$ |
| all-frames: full L-shape solve | converges at step 85, the same step as the classical solver; $\max\lvert A_z^q-A_z^c\rvert = 3\cdot10^{-13}$ |
| ctrl-π opt: one step, random $\nu$ per cell, random fields and $J$ | max error $\le 1.6\cdot10^{-15}$; MAP + UNMAP 300 CX, down from 5040 |
| notebook: ctrl-π opt at $q=3$ (28 qubits), first 5 steps | matches the classical iterates to $\sim10^{-16}$ |

## Notes

- **Sign of the source.** All codes use $+J$, as in the paper's 3-D update. Equivalently, $(\Delta - S)\,a = J$ is the discrete curl-curl system.
- **Damping.** For a genuinely 3-D problem (all three components nonzero), damped Jacobi is stable only for $\beta < 2/\lambda_{\max}(D^{-1}M) \approx 0.67$. The L-shape benchmark uses $\beta = 0.8$. That works only because $A_x = A_y = 0$ there, which makes the problem effectively 2-D.
- **What is idealised.** The encoding uses `set_statevector` (an Aer instruction; on hardware this would be `StatePreparation` or an oracle), and the readout uses exact amplitudes (on hardware, amplitude estimation).
- Some of the explanatory `.md` notes refer to scripts and notebooks from the full research folder that are not part of this upload.
