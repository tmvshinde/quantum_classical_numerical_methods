# The star shifter, explained simply

This note explains the quantum circuit in `star_shifter.py`, step by step. It assumes you know what a qubit, a basis state and a CNOT gate are. Nothing more.

---

## 1. What problem are we solving?

In our simulation, every edge of the grid has an **address** $p = (k, i, j)$ (three whole numbers, one per direction). To update an edge, we need the values of its **neighbours**: the edge one step to the right, one step to the left, one step up, and so on.

So we need an operation that moves data from address $p+\delta$ to address $p$, for a few small offsets $\delta$ like $(0, +1, 0)$ or $(+1, -1, 0)$. That is a **shift**.

On a classical computer you shift an array by copying every entry, one by one. That takes as many steps as there are grid points.

On a quantum computer the addresses are stored in qubits, and the data sits in the **amplitudes** of the state:

$$
|\psi\rangle = \sum_p \psi[p]\,|p\rangle .
$$

A circuit that changes the address, $|p\rangle \to |p+\delta\rangle$, moves **every** amplitude at the same time, because quantum gates act on all basis states at once (linearity):

$$
\sum_p \psi[p]\,|p\rangle \;\longrightarrow\; \sum_p \psi[p]\,|p+\delta\rangle .
$$

So we only need a circuit that adds a small number to an address. The rest happens automatically.

---

## 2. Adding 1 to a binary number

An address along one direction is a number between $0$ and $2^q-1$, stored in $q$ qubits: $g_0$ is the lowest bit, $g_{q-1}$ the highest. With $q = 3$ we can count from 0 to 7.

**Adding 1 by hand** works exactly as in school, with carries:

```
     0 1 1      (= 3)
   +     1
   -------
     1 0 0      (= 4)
```

- Bit 0 flips (1 → 0) and produces a carry.
- The carry flips bit 1 (1 → 0) and produces another carry.
- That carry flips bit 2 (0 → 1). Done.

**The rule:** bit $m$ flips exactly when all the bits below it are 1. We call that condition the **carry** $c_m$:

$$
c_0 = 1, \qquad c_1 = g_0, \qquad c_2 = g_0 \text{ AND } g_1, \qquad \dots
$$

A useful special case: $7 + 1 = 111 + 1 = 000$. The number **wraps around** to 0. That is called a *cyclic* or *periodic* shift.

---

## 3. The increment circuit

On a quantum computer, "AND" is a **Toffoli gate**: it flips a target qubit if both control qubits are 1. We store each carry in an extra qubit (an **ancilla**), which starts in $|0\rangle$.

The circuit has two halves:

```
 1. compute the carries, going UP          2. flip the bits and clean up, going DOWN
    c0 = cin AND g0                            g2 ^= c1   then erase c1
    c1 = c0  AND g1                            g1 ^= c0   then erase c0
                                               g0 ^= cin
```

($\wedge$ is AND, and "^=" means "flip if the right-hand side is 1", i.e. a CNOT.)

**Why erase the carries?** The ancillas must end in $|0\rangle$ again. Otherwise they stay entangled with the data, and the shift would not be clean. We erase each carry with the same Toffoli that created it. This only works while its input bits are still unchanged, which is why we go top-down: we flip $g_2$ first, then erase $c_1$ (which needs $g_1$, not flipped yet), and so on.

**Worked example**, $x = 3 = 011$ (so $g_0=1, g_1=1, g_2=0$), with $\text{cin}=1$:

| step | operation | $g_2 g_1 g_0$ | $c_1\,c_0$ |
|---|---|---|---|
| start | | 0 1 1 | 0 0 |
| 1 | $c_0 = \text{cin}\wedge g_0 = 1$ | 0 1 1 | 0 1 |
| 2 | $c_1 = c_0\wedge g_1 = 1$ | 0 1 1 | 1 1 |
| 3 | $g_2$ ^= $c_1$ | **1** 1 1 | 1 1 |
| 4 | erase $c_1$ (uses $g_1 = 1$, still unchanged) | 1 1 1 | 0 1 |
| 5 | $g_1$ ^= $c_0$ | 1 **0** 1 | 0 1 |
| 6 | erase $c_0$ (uses $g_0 = 1$, still unchanged) | 1 0 1 | 0 0 |
| 7 | $g_0$ ^= cin | 1 0 **0** | 0 0 |

The result is $100 = 4$, and the carries are back to 0. ✓

Picture of this circuit for $q = 4$: `circuits/01_controlled_increment_q4.png`.

---

## 4. Trick 1: the control is free

Often we want "add 1 **only if** some flag is 1". The trick is to put the flag in as the **carry-in** $\text{cin}$, the very first input of the chain:

- if $\text{cin} = 1$, the circuit adds 1 (as above);
- if $\text{cin} = 0$, every carry is 0, so nothing flips: the circuit adds 0.

So a **controlled** increment costs the same as a normal one: $q-1$ Toffolis for the carries.

---

## 5. Trick 2: subtracting 1 using "add 1"

To subtract 1 we don't need a new circuit. We use this identity:

$$
x - 1 = \text{NOT}\big(\text{NOT}(x) + 1\big),
$$

where NOT flips every bit.

**Example**, $x = 4 = 100$:

```
NOT(100) = 011      (= 3)
011 + 1  = 100      (= 4)
NOT(100) = 011      (= 3)  =  4 - 1   ✓
```

On qubits, "NOT every bit" is one X gate per bit. If we want to choose between $+1$ and $-1$ with a qubit $\sigma$, we use **CNOTs from $\sigma$** instead of X gates. Then:

- $\sigma = 0$: no flips, so the circuit adds 1;
- $\sigma = 1$: NOT, +1, NOT, so the circuit subtracts 1.

CNOTs are cheap, so the direction costs almost nothing. Picture: `circuits/02_controlled_pm_shift_q4.png`.

---

## 6. The star: 12 shifts, one circuit

To update one edge in our curl-curl stencil we need **12** neighbour terms (the edge's "star"). Their offsets $(\delta k, \delta i, \delta j)$ are:

| face | term `a` | term `bc_k` | term `bc_k1` |
|---|---|---|---|
| E (east) | $(0, +1, 0)$ | $(0, 0, 0)$ | $(+1, 0, 0)$ |
| W (west) | $(0, -1, 0)$ | $(0, -1, 0)$ | $(+1, -1, 0)$ |
| N (north) | $(0, 0, +1)$ | $(0, 0, 0)$ | $(+1, 0, 0)$ |
| S (south) | $(0, 0, -1)$ | $(0, 0, -1)$ | $(+1, 0, -1)$ |

Looking at the table, three facts make the circuit simple:

1. $\delta k$ is only ever 0 or $+1$.
2. $\delta i$ and $\delta j$ are only ever $-1$, 0 or $+1$.
3. No term moves in both $i$ and $j$.

### The term register

We add 4 qubits that say **which term** we are working on:

| qubit | meaning |
|---|---|
| $\alpha$ | which axis: 0 = E/W (the $i$-axis), 1 = N/S (the $j$-axis) |
| $\sigma$ | which side: 0 = E or N (the "+" side), 1 = W or S (the "−" side) |
| $r_1, r_0$ | which member: `a` = 01, `bc_k` = 00, `bc_k1` = 10 |

These codes were **chosen on purpose**, so that the information the circuit needs can be read off almost directly:

| what the circuit needs | how it is obtained | cost |
|---|---|---|
| "add 1 to $k$?" | it is just $r_1$ | free |
| "shift along the axis at all?" | $s = \sigma$ OR $r_0$ | 1 Toffoli |
| "is it the $i$-axis?" | $f_i = s$ AND (NOT $\alpha$) | 1 Toffoli |
| "is it the $j$-axis?" | $f_j = s$ XOR $f_i$ | 2 CNOTs |
| "minus direction?" | it is just $\sigma$ | free |

### The whole circuit

```
FLAGS        compute s, f_i, f_j from the term register
k-axis       +1 with carry-in r1
i-axis       NOT^σ,  +1 with carry-in f_i,  NOT^σ        (= ±1 if f_i)
j-axis       NOT^σ,  +1 with carry-in f_j,  NOT^σ        (= ±1 if f_j)
UNFLAG       undo the flag computation (flags back to 0)
```

The carry qubits are shared: each of the three increments cleans them up before the next one starts. Pictures: `circuits/03b_star_shifter_q3_blocks.png` (boxes) and `circuits/03_star_shifter_q3.png` (gates).

### Example: the term W_bc_k1

Code: $\alpha = 0$, $\sigma = 1$, $r_1 = 1$, $r_0 = 0$.

- $r_1 = 1$, so $k$ gets $+1$.
- $s = \sigma \text{ OR } r_0 = 1$; $f_i = s \text{ AND NOT }\alpha = 1$; $f_j = 0$.
- $\sigma = 1$, so the $i$-shift is a $-1$.

The offset is $(+1, -1, 0)$, which matches the table. ✓

### All 12 terms at once

If the term register is in a **superposition** of all 12 codes, the circuit shifts each branch by its own offset, in a single run:

$$
\frac{1}{\sqrt{12}}\sum_\tau |\tau\rangle \otimes \sum_p \psi[p]\,|p\rangle
\;\longrightarrow\;
\frac{1}{\sqrt{12}}\sum_\tau |\tau\rangle \otimes \sum_p \psi[p]\,|p+\delta_\tau\rangle .
$$

That is the main point: the circuit costs about **three increments**, no matter how many stencil terms there are.

---

## 7. How do we know it works?

Three tests, all passing:

- **`star_shifter.py`**: a classical bit-by-bit check of every input (all 12 terms × all addresses) for $q = 1, 2, 3, 4$. The output address is $p+\delta$, and all helper qubits are back to 0.
- **`test_star_shifter_aer.py`**: a random quantum state is loaded, the circuit is run in Qiskit's Aer simulator, and the result is compared with the classically shifted array. For every term the overlap is exactly 1. In the printout you can see each amplitude land at its new address unchanged, e.g. term E_a moves $|2\,5\,3\rangle \to |2\,6\,3\rangle$.
- **`L_shape_mu_3d_quantum_shift.ipynb`**: the shifter is used inside a full magnetostatics solver (the L-shaped iron object). It gives the same answer as the classical solver to about $10^{-13}$.

---

## 8. How expensive is it?

With $q$ qubits per direction (so a grid of $2^q \times 2^q \times 2^q$ points):

| | count |
|---|---|
| Toffoli gates | $3q - 1$ to compute (plus the same number to erase) |
| qubits | $3q$ address + 4 term + 3 flags + $(q-1)$ carries $= 4q + 6$ |

Both grow like $\log(\text{number of grid points})$. Doubling the grid in every direction adds only about 3 Toffolis.

For comparison: the textbook way to add 1 (one big multi-controlled gate per bit) costs about $q^2$ gates per direction. Doing each of the 12 terms separately would cost about 12 times more again.

---

## 9. Words used in this note

| word | meaning |
|---|---|
| **amplitude** | the complex number in front of a basis state, e.g. $\psi[p]$ in $\psi[p]\,\lvert p\rangle$ |
| **ancilla** | a helper qubit that starts in $\lvert0\rangle$ and must end in $\lvert0\rangle$ |
| **carry** | the "1 to carry over" when adding binary numbers |
| **CNOT** | flips the target qubit if the control qubit is 1 |
| **Toffoli** | flips the target qubit if both control qubits are 1 (a quantum AND) |
| **cyclic shift** | adding modulo $2^q$, so the highest address wraps around to 0 |
| **superposition** | a sum of several basis states at once; a gate acts on all of them in one go |
| **term register** | the 4 qubits that say which of the 12 stencil terms a branch belongs to |
