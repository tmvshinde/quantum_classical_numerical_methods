# L_shape_two_methods: the L-shaped iron benchmark with both quantum circuits

The problem:
- an L-shaped object with $\mu_r = 10$ on an $8^3$ grid;
- $A_z = 10^{-2}$ at $x=0$ and $A_z = 5\cdot10^{-2}$ at $x=1$;
- $\partial_y A_z = 0$ at $y = 0, 1$;
- $J = 0$ and $\beta = 0.8$.

The notebook [`L_shape_two_methods.ipynb`](L_shape_two_methods.ipynb) runs:

| Method | Qubits | Used for |
|---|---|---|
| classical (blocks + slices) | – | reference |
| **all-frames** circuit | 23 | **full solve** to convergence (85 steps) |
| **ctrl-π opt** circuit, rebuilt for $q = 3$ | 28 | the **first 5 steps**, compared step by step with the classical iterates (≈2–3 min per step) |

It then computes $\mathbf B = \nabla\times\mathbf A$ on the faces and plots $A_z$ and $\lvert\mathbf B\rvert$ (with field lines) on the mid z-plane.

All files apart from the notebook are copies of the modules from the other two folders. The notebook imports them.

```bash
jupyter notebook L_shape_two_methods.ipynb
```

The notebook writes:
- `results/L_shape_two_methods/` — the $a_x, a_y, a_z$ text files (classical, all-frames, ctrl-π step 5) and `A_B_midplane.svg`;
- `circuits/L_shape_two_methods/` — the circuit SVGs.

The 28-qubit statevector needs about 4 GB of RAM.
