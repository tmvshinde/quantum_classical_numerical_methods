# all_frames_weighted: one Jacobi step for all three frames in one circuit

Each frame (z, y, x) is encoded in its **own** index layout, so the 12 star offsets are the same in every frame. A 2-qubit `frame` register labels the components and keeps them apart.

Each frame's state has three subspaces, set by the `type` register:

| type | content |
|---|---|
| 0 | the 12 star product blocks, divided by $\Delta$ at the target edge |
| 1 | $J/\Delta$ |
| 2 | $a$ |

The circuit is:

```
SIGN star  ->  STAR SHIFTER^dagger  ->  beta / (1 - beta) on a weight ancilla  ->  H on term + type
```

The amplitude at $\lvert \text{frame}=f,\ \text{rest}=0,\ p\rangle$, times $8\mathcal N$, is $a^{\text{new}}_f(p)$.

The circuit has 23 qubits for the $8^3$ grid: 18 for the star shifter, 2 for `type`, 1 for the weight ancilla and 2 for `frame`.

| File | |
|---|---|
| `test_jacobi_all_frames_weighted.py` | the circuit, one-step check against the classical update, full L-shape solve, circuit saving |
| `test_jacobi_all_frames_weighted.md` | detailed explanation: registers, **term codes**, encoding, stages, readout, results |
| `star_shifter.py`, `star_shifter_explained.md`, `star_term_shifts.md` | the star shifter and how the 12 terms become shifts of product blocks |
| `test_combined_frames_aer.py` | L-shape problem, faces, blocks, frame layouts (imported) |
| `circuit_saving.py` | SVG / QPY saving |

```bash
python test_jacobi_all_frames_weighted.py
```

Writes `circuits/jacobi_all_frames_weighted/*.svg`.
