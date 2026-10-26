# ctrl_pi_opt: one encoding, all frames by controlled-π, structured labels

The 12 frame-independent **edge–face products** $A_d\cdot F_c$, the sources $J_x, J_y, J_z$ and $a_d\cdot\Delta_d$ are encoded **once**, in the storage layout $(z,x,y)$, under a 5-qubit block label $L$. Inside the circuit:

1. **FRAME** puts the frame register into $(\lvert z\rangle+\lvert y\rangle+\lvert x\rangle)/\sqrt3$, and **COPY** duplicates the products that one frame uses twice.
2. **ctrl-π**: controlled SWAPs rotate the index groups into each frame's layout.
3. **MAP** writes the star-term code of the role each block plays in that frame. The **STAR SHIFTER†** then shifts every branch, and **UNMAP** clears the term code again.
4. **W1** multiplies by $1/\Delta_f(p)$ at the target edge. **W2** applies the constants: $\pm\beta$ for the star terms, $+\beta$ for $J$, and $1-\beta$ for $a\Delta$.
5. **SUM**: Hadamards on the label and copy registers add all the terms.

**Optimisation (`test_jacobi_ctrl_pi_opt.py`).** The label is structured as $L = s + 2h + 4d + 16\,\text{kind}$:

| part | meaning |
|---|---|
| $d$ | edge component |
| $h$ | which face normal |
| $s$ | face slot |
| kind | product, or $J$ / $a\Delta$ |

A frame only changes the edge's role, $r = d - k_f \pmod 3$. This gives:
- **MAP-A** is a controlled mod-3 step on $d$;
- **MAP-B** is 8 small MCX gates (at most 4 controls);
- **W2** is a 64-entry table.

MAP + UNMAP drops from 5040 CX (120 MCX with up to 8 controls) to **300 CX**.

The test case is 24 qubits on a $4^3$ grid ($q=2$), with a random $\nu$ in every cell, which makes every face weight different. One run gives all three frames, with error $\le 1.6\cdot10^{-15}$ against the classical update.

| File | |
|---|---|
| `test_jacobi_ctrl_pi_opt.py` | structured labels, MAP-A/MAP-B, W2, one-step check, gate counts, results and circuits |
| `test_jacobi_ctrl_pi.py`, `test_jacobi_ctrl_pi.md` | the base ctrl-π construction (encoding, ctrl-π, W1, readout) and its explanation |
| `star_shifter.py`, `star_shifter_explained.md` | the 12-term star shifter |
| `test_combined_frames_aer.py` | product blocks and term table (imported) |
| `circuit_saving.py` | SVG / QPY saving |

```bash
python test_jacobi_ctrl_pi_opt.py
```

This writes:
- `results/jacobi_ctrl_pi_opt/{ax,ay,az}_{quantum,classical}.txt`, in shape-preserving text;
- `circuits/jacobi_ctrl_pi_opt/*.svg`.
