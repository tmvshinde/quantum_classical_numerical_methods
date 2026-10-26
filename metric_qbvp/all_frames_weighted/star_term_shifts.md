# Star terms as shifts of edge–face product blocks

This note answers one question: **how do you get every term of the material curl-curl stencil by encoding products first and then shifting?**

The rule for which product to shift, and by how much, is given for all three frames below. Every entry was checked numerically against the classical material solver (see *Verification*).

---

## 0. Material data: where $F_x, F_y, F_z$ and `nfy`, `nfx` come from

### 0.1 Cell reluctivity

The material is given per **cell** (3-cell) of the grid, in physical $(x,y,z)$ layout:

```python
nu_cell = 1.0 / mu_cell          # shape (nx-1, ny-1, nz-1)
```

For example, for the L-shaped object, `nu_cell = np.where(L_mask, 1/mu_r, 1.0)`.

### 0.2 Face families $F_x, F_y, F_z$

The curl-curl operator is $M = \mathbf D^{\mathsf T}\mathbf H\mathbf D$. Here $\mathbf D$ maps edge values $A$ to face fluxes $B = \operatorname{curl} A$, and the Hodge operator $\mathbf H$ is **diagonal on the faces**: $H_f = \nu_f\,|\tilde\omega_f|/|\omega_f|$, where $\tilde\omega_f$ is the dual edge crossing face $f$.

The dual edge of a face runs between the centres of the **two cells on either side of the face**, half of it in each cell. So the face reluctivity is the length-weighted average:

$$
H_f = \frac{\nu_{1}\,\ell_1 + \nu_{2}\,\ell_2}{|\omega_f|},
\qquad \ell_1 = \ell_2 = \tfrac{h}{2}\ \text{(uniform grid)}
\ \Rightarrow\ H_f = \frac{h}{h^2}\cdot\tfrac12(\nu_1 + \nu_2).
$$

On a uniform grid the geometric factor $h/h^2 = 1/h$ is the same for every face, and it is absorbed into the normalisation of the stencil and the source. What remains is the **average of the two cells across the face**.

Group the faces by their normal direction $c \in \{x, y, z\}$. A face of normal $c$ at index $p$ lies between cells $p - e_c$ and $p$:

$$
F_c[p] = \tfrac12\big(\nu_{\text{cell}}[p - e_c] + \nu_{\text{cell}}[p]\big).
$$

| Family | Faces | Cells averaged | Shape $(x,y,z)$ |
|---|---|---|---|
| $F_x$ | normal $x$ (the $yz$-faces) | $[i-1,j,k]$ and $[i,j,k]$ | $(n_x,\ n_y-1,\ n_z-1)$ |
| $F_y$ | normal $y$ (the $zx$-faces) | $[i,j-1,k]$ and $[i,j,k]$ | $(n_x-1,\ n_y,\ n_z-1)$ |
| $F_z$ | normal $z$ (the $xy$-faces) | $[i,j,k-1]$ and $[i,j,k]$ | $(n_x-1,\ n_y-1,\ n_z)$ |

```python
def face_family(nu, axis):
    """nu on the faces normal to 'axis': average of the two cells across each face.
    nu: cell reluctivity, (x, y, z) layout."""
    nu = np.moveaxis(nu, axis, 0)
    f = np.empty((nu.shape[0] + 1,) + nu.shape[1:])
    f[1:-1] = 0.5*(nu[:-1] + nu[1:])     # interior faces: two cells
    f[0], f[-1] = nu[0], nu[-1]          # boundary faces: one cell (see note)
    return np.moveaxis(f, 0, axis)

Fx = face_family(nu_cell, 0)
Fy = face_family(nu_cell, 1)
Fz = face_family(nu_cell, 2)
```

**Boundary faces.** On the domain boundary only one cell exists. The interior update never reads these values: every interior edge's four faces are interior along their normal. So the code simply copies the adjacent cell. A consistent DEC weight would be $\tfrac12\nu_{\text{cell}}$ (only half of the dual edge lies inside the domain). That matters if natural (Neumann-type) boundary edges are updated.

**Non-uniform or deformed grids.** Use the full expression $H_f = (\nu_1\ell_1 + \nu_2\ell_2)/|\omega_f|$ instead of the plain average. Everything else in this note is unchanged, because the metric enters only through these face values.

### 0.3 Per-frame face arrays `nfy` and `nfx`

The stencil of each frame needs two face families:
- `nfy`: faces whose normal is the frame's local **j** axis (the E/W faces);
- `nfx`: faces whose normal is the frame's local **i** axis (the N/S faces).

| Frame | Layout $(k,i,j)$ | `nfy` | `nfx` | Permutation from $(x,y,z)$ |
|---|---|---|---|---|
| z | $(z, x, y)$ | $F_y$ | $F_x$ | `a.transpose(2, 0, 1)` |
| y | $(y, z, x)$ | $F_x$ | $F_z$ | `a.transpose(1, 2, 0)` |
| x | $(x, y, z)$ | $F_z$ | $F_y$ | identity |

```python
to_z = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1))   # (x,y,z) -> (z,x,y)
to_y = lambda a: np.ascontiguousarray(a.transpose(1, 2, 0))   # (x,y,z) -> (y,z,x)

nfy_z, nfx_z = to_z(Fy), to_z(Fx)     # z-frame
nfy_y, nfx_y = to_y(Fx), to_y(Fz)     # y-frame
nfy_x, nfx_x = Fz,       Fy           # x-frame
```

In frame coordinates:
- `nfy[k,i,j]` is the face spanning $i \to i+1$ and $k \to k+1$ at position $j$. It lies between cells `[k,i,j-1]` and `[k,i,j]`.
- `nfx[k,i,j]` is the face spanning $j \to j+1$ and $k \to k+1$ at position $i$. It lies between cells `[k,i-1,j]` and `[k,i,j]`.

### 0.4 Link to `_coefficients`

The four coefficients of an edge are lookups into these arrays:

| `_coefficients` (from the cell array) | Face-array lookup |
|---|---|
| `nE = 0.5*(nu[k,i,j-1] + nu[k,i,j])` | `nfy[k,i,j]` |
| `nW = 0.5*(nu[k,i-1,j-1] + nu[k,i-1,j])` | `nfy[k,i-1,j]` |
| `nN = 0.5*(nu[k,i-1,j] + nu[k,i,j])` | `nfx[k,i,j]` |
| `nS = 0.5*(nu[k,i-1,j-1] + nu[k,i,j-1])` | `nfx[k,i,j-1]` |
| `D = nE + nW + nN + nS` | sum of the four |

Here `nu` is the cell array **in the frame's layout**: `nuz = to_z(nu_cell)`, `nuy = to_y(nu_cell)`, `nux = nu_cell`.

**Two equivalent routes.** Either compute $F_x, F_y, F_z$ once and permute them into each frame (table above), or permute `nu_cell` into each frame and call `face_nu` / `_coefficients` there. Both give identical arrays.

**What does not work** is computing nE…D in one frame and permuting *those* arrays into the other frames. The coefficients are frame-dependent: an x-edge uses $F_z, F_y$, a y-edge uses $F_x, F_z$. They only coincide if $\nu$ itself is unchanged by the rotation $x\to y\to z$ (see `test_coefficient_permutation.py`).

### 0.5 Under the cyclic rotation π

$F_x, F_y, F_z$ are labelled by a direction (the face normal), so they transform like $(A_x, A_y, A_z)$:

$$
\pi:\ F_x \to F_y \to F_z \to F_x \quad\text{(plus the index rotation } (z,x,y)\to(y,z,x)\text{)}.
$$

After $\pi^s$, the canonical slots `nfy` = $F_y$ slot and `nfx` = $F_x$ slot hold the correct families for frame $s$.

On the quantum side, a **single Hodge oracle** $O_H(c, p)$ in physical coordinates serves all frames when conjugated by π. The family label becomes $c = (c_{\text{canonical}} - s) \bmod 3$, and the index is rotated back by $\pi^{-s}$ before the material map (e.g. the L-mask comparators) is evaluated. See `test_cyclic_material.py`.

---

## 1. The terms to produce

For a z-edge at index $p$ (the z-frame), the off-diagonal part of the stencil is the sum of four face contributions:

```
E:  F_y[p]       · ( A_z[p+e_x]  +  A_x[p]        −  A_x[p+e_z]       )
W:  F_y[p−e_x]   · ( A_z[p−e_x]  −  A_x[p−e_x]    +  A_x[p−e_x+e_z]   )
N:  F_x[p]       · ( A_z[p+e_y]  +  A_y[p]        −  A_y[p+e_z]       )
S:  F_x[p−e_y]   · ( A_z[p−e_y]  −  A_y[p−e_y]    +  A_y[p−e_y+e_z]   )
```

The Jacobi update is then

$$
A_z[p] \leftarrow \frac{\beta}{\Delta[p]}\big(E + W + N + S + J_z[p]\big) + (1-\beta)\,A_z[p],
\qquad \Delta[p] = F_y[p] + F_y[p-e_x] + F_x[p] + F_x[p-e_y].
$$

This is exactly `_relax` with $(k,i,j) = (z,x,y)$, $a = A_z$, $b = A_x$, $c = A_y$, nE = `nfy[k,i,j]`, nW = `nfy[k,i-1,j]`, nN = `nfx[k,i,j]` and nS = `nfx[k,i,j-1]`.

---

## 2. Why "multiply then shift" needs two products per face family

A shift (a translation of the index register) moves the **whole amplitude**, so the face ν and the field move together:

| Start from | Shift | Result | Needed? |
|---|---|---|---|
| $F_z[p]\,A_x[p]$ | $-e_y$ | $F_z[p-e_y]\,A_x[p-e_y]$ | ✓ |
| $F_z[p]\,A_x[p]$ | $+e_y$ | $F_z[p+e_y]\,A_x[p+e_y]$ | ✗ wrong face |
| $F_z[p-e_y]\,A_x[p]$ | $+e_y$ | $F_z[p]\,A_x[p+e_y]$ | ✓ |

Once F and A are multiplied into one amplitude, no unitary can separate them again. Index permutations keep them at the same index, and linear combinations only add such products. So **each edge must be multiplied by both of its faces of each family**, before any shifting.

---

## 3. The encoded blocks $Q$

Every edge $A_d[p]$ lies on four faces: two of each normal $c \neq d$. Let $t$ be the third direction ($t \neq c, d$). The two faces of normal $c$ are at $p$ (slot **0**) and $p - e_t$ (slot **−**).

$$
Q[d;\,c,\,0][p] = F_c[p]\,A_d[p], \qquad Q[d;\,c,\,-][p] = F_c[p-e_t]\,A_d[p].
$$

That gives **12 blocks**, all with the shape of the edge array $A_d$:

| Edge | Slot 0 (face at $p$) | Slot − (face at $p - e_t$) |
|---|---|---|
| $A_x$ | $Q[x;y,0] = F_y[p]\,A_x[p]$ | $Q[x;y,-] = F_y[p-e_z]\,A_x[p]$ |
| | $Q[x;z,0] = F_z[p]\,A_x[p]$ | $Q[x;z,-] = F_z[p-e_y]\,A_x[p]$ |
| $A_y$ | $Q[y;z,0] = F_z[p]\,A_y[p]$ | $Q[y;z,-] = F_z[p-e_x]\,A_y[p]$ |
| | $Q[y;x,0] = F_x[p]\,A_y[p]$ | $Q[y;x,-] = F_x[p-e_z]\,A_y[p]$ |
| $A_z$ | $Q[z;x,0] = F_x[p]\,A_z[p]$ | $Q[z;x,-] = F_x[p-e_y]\,A_z[p]$ |
| | $Q[z;y,0] = F_y[p]\,A_z[p]$ | $Q[z;y,-] = F_y[p-e_x]\,A_z[p]$ |

The sources $J_x, J_y, J_z$ are encoded alongside, unchanged.

---

## 4. Reading rule

A star term $F_c[f]\cdot A_d[q]$ is read from block $Q[d;\,c,\,s]$ at index $q$:

- **slot $s = 0$** if the face index equals the edge index ($f = q$);
- **slot $s = -$** if $f = q - e_t$.

The **offset** is $\delta = q - p$: the term needed at $p$ is the value of the block at $p + \delta$. In the circuit this is a translation of the index register by $-\delta$, which moves the amplitude at $p+\delta$ to $p$.

A useful pattern: **slot 0 is read at offset $0$ or $-e$, and slot − is read at offset $+e$.** Combined offsets such as $-e_x + e_z$ also occur.

---

## 5. Shift tables

The **Sign** column is the sign of the term inside its face bracket.

### z-frame: updating $A_z[p]$

| Face | Sign | Term | Block | Offset $\delta$ |
|---|---|---|---|---|
| E | + | $F_y[p]\,A_z[p+e_x]$ | $Q[z;y,-]$ | $+e_x$ |
| E | + | $F_y[p]\,A_x[p]$ | $Q[x;y,0]$ | $0$ |
| E | − | $F_y[p]\,A_x[p+e_z]$ | $Q[x;y,-]$ | $+e_z$ |
| W | + | $F_y[p-e_x]\,A_z[p-e_x]$ | $Q[z;y,0]$ | $-e_x$ |
| W | − | $F_y[p-e_x]\,A_x[p-e_x]$ | $Q[x;y,0]$ | $-e_x$ |
| W | + | $F_y[p-e_x]\,A_x[p-e_x+e_z]$ | $Q[x;y,-]$ | $-e_x+e_z$ |
| N | + | $F_x[p]\,A_z[p+e_y]$ | $Q[z;x,-]$ | $+e_y$ |
| N | + | $F_x[p]\,A_y[p]$ | $Q[y;x,0]$ | $0$ |
| N | − | $F_x[p]\,A_y[p+e_z]$ | $Q[y;x,-]$ | $+e_z$ |
| S | + | $F_x[p-e_y]\,A_z[p-e_y]$ | $Q[z;x,0]$ | $-e_y$ |
| S | − | $F_x[p-e_y]\,A_y[p-e_y]$ | $Q[y;x,0]$ | $-e_y$ |
| S | + | $F_x[p-e_y]\,A_y[p-e_y+e_z]$ | $Q[y;x,-]$ | $-e_y+e_z$ |

### y-frame: updating $A_y[p]$

This is the z-frame table relabelled with $z\to y$, $x\to z$, $y\to x$.

| Face | Sign | Term | Block | Offset $\delta$ |
|---|---|---|---|---|
| E | + | $F_x[p]\,A_y[p+e_z]$ | $Q[y;x,-]$ | $+e_z$ |
| E | + | $F_x[p]\,A_z[p]$ | $Q[z;x,0]$ | $0$ |
| E | − | $F_x[p]\,A_z[p+e_y]$ | $Q[z;x,-]$ | $+e_y$ |
| W | + | $F_x[p-e_z]\,A_y[p-e_z]$ | $Q[y;x,0]$ | $-e_z$ |
| W | − | $F_x[p-e_z]\,A_z[p-e_z]$ | $Q[z;x,0]$ | $-e_z$ |
| W | + | $F_x[p-e_z]\,A_z[p+e_y-e_z]$ | $Q[z;x,-]$ | $+e_y-e_z$ |
| N | + | $F_z[p]\,A_y[p+e_x]$ | $Q[y;z,-]$ | $+e_x$ |
| N | + | $F_z[p]\,A_x[p]$ | $Q[x;z,0]$ | $0$ |
| N | − | $F_z[p]\,A_x[p+e_y]$ | $Q[x;z,-]$ | $+e_y$ |
| S | + | $F_z[p-e_x]\,A_y[p-e_x]$ | $Q[y;z,0]$ | $-e_x$ |
| S | − | $F_z[p-e_x]\,A_x[p-e_x]$ | $Q[x;z,0]$ | $-e_x$ |
| S | + | $F_z[p-e_x]\,A_x[p-e_x+e_y]$ | $Q[x;z,-]$ | $-e_x+e_y$ |

### x-frame: updating $A_x[p]$

This is the z-frame table relabelled with $z\to x$, $x\to y$, $y\to z$.

| Face | Sign | Term | Block | Offset $\delta$ |
|---|---|---|---|---|
| E | + | $F_z[p]\,A_x[p+e_y]$ | $Q[x;z,-]$ | $+e_y$ |
| E | + | $F_z[p]\,A_y[p]$ | $Q[y;z,0]$ | $0$ |
| E | − | $F_z[p]\,A_y[p+e_x]$ | $Q[y;z,-]$ | $+e_x$ |
| W | + | $F_z[p-e_y]\,A_x[p-e_y]$ | $Q[x;z,0]$ | $-e_y$ |
| W | − | $F_z[p-e_y]\,A_y[p-e_y]$ | $Q[y;z,0]$ | $-e_y$ |
| W | + | $F_z[p-e_y]\,A_y[p+e_x-e_y]$ | $Q[y;z,-]$ | $+e_x-e_y$ |
| N | + | $F_y[p]\,A_x[p+e_z]$ | $Q[x;y,-]$ | $+e_z$ |
| N | + | $F_y[p]\,A_z[p]$ | $Q[z;y,0]$ | $0$ |
| N | − | $F_y[p]\,A_z[p+e_x]$ | $Q[z;y,-]$ | $+e_x$ |
| S | + | $F_y[p-e_z]\,A_x[p-e_z]$ | $Q[x;y,0]$ | $-e_z$ |
| S | − | $F_y[p-e_z]\,A_z[p-e_z]$ | $Q[z;y,0]$ | $-e_z$ |
| S | + | $F_y[p-e_z]\,A_z[p+e_x-e_z]$ | $Q[z;y,-]$ | $+e_x-e_z$ |

**Observations.**
- Each frame uses **8 of the 12 blocks**. The z-frame never needs $Q[x;z,\cdot]$ or $Q[y;z,\cdot]$, because faces of normal z don't contain z-edges.
- The four blocks of the edge's **own** direction are each read once (at $\pm e$). The four **cross** blocks are each read twice (e.g. $Q[x;y,0]$ at $0$ and $-e_x$).
- Offsets have at most two non-zero components. So each term needs at most two ±1 translations, on different index groups.

### Diagonal term

The four "self" products of the edge are the blocks read at offset 0 for the edge's own direction:

$$
\Delta[p]\,A_z[p] = Q[z;y,0][p] + Q[z;y,-][p] + Q[z;x,0][p] + Q[z;x,-][p].
$$

For the Jacobi form you only need the scalar $\Delta[p]$, the sum of the 4 face ν values. The Hodge oracle provides it by summing the edge's four faces, so no shift is needed. The diagonal products are only needed if you use the residual form $A - \beta\Delta^{-1}(MA - J)$.

---

## 5b. All three frames in local index coordinates $(k, i, j)$

In each frame's own layout, the three frames are **one table**. Only the meaning of the symbols changes.

### Frame identification

| Frame | $(k, i, j)$ | $a$ (updated) | $b$ | $c$ | `nfy` (E/W faces) | `nfx` (N/S faces) |
|---|---|---|---|---|---|---|
| z | $(z, x, y)$ | $A_z$ | $A_x$ | $A_y$ | $F_y$ | $F_x$ |
| y | $(y, z, x)$ | $A_y$ | $A_z$ | $A_x$ | $F_x$ | $F_z$ |
| x | $(x, y, z)$ | $A_x$ | $A_y$ | $A_z$ | $F_z$ | $F_y$ |

All arrays are in the frame layout: `a = azz, azy, azx`, `b = axz, axy, axx`, `c = ayz, ayy, ayx`, and the face arrays are permuted with the same transforms (`zxy_to_yzx`, `zxy_to_xyz`).

### Local blocks (edge-aligned products)

| Block | Definition | Face |
|---|---|---|
| $Q_a^{y0}[k,i,j]$ | `nfy[k,i,j] · a[k,i,j]` | E face of $a$ |
| $Q_a^{y-}[k,i,j]$ | `nfy[k,i-1,j] · a[k,i,j]` | W face of $a$ |
| $Q_a^{x0}[k,i,j]$ | `nfx[k,i,j] · a[k,i,j]` | N face of $a$ |
| $Q_a^{x-}[k,i,j]$ | `nfx[k,i,j-1] · a[k,i,j]` | S face of $a$ |
| $Q_b^{y0}[k,i,j]$ | `nfy[k,i,j] · b[k,i,j]` | upper `nfy` face of $b$ |
| $Q_b^{y-}[k,i,j]$ | `nfy[k-1,i,j] · b[k,i,j]` | lower `nfy` face of $b$ |
| $Q_c^{x0}[k,i,j]$ | `nfx[k,i,j] · c[k,i,j]` | upper `nfx` face of $c$ |
| $Q_c^{x-}[k,i,j]$ | `nfx[k-1,i,j] · c[k,i,j]` | lower `nfx` face of $c$ |

### Shift table (the same for every frame)

Offsets are $(\delta k, \delta i, \delta j)$: the block is read at $[k+\delta k,\ i+\delta i,\ j+\delta j]$.

| Face | Sign | Term (as in `_relax`) | Block | Read at | Offset |
|---|---|---|---|---|---|
| E | + | `nfy[k,i,j] · a[k,i+1,j]` | $Q_a^{y-}$ | $[k,\ i+1,\ j]$ | $(0,+1,0)$ |
| E | + | `nfy[k,i,j] · b[k,i,j]` | $Q_b^{y0}$ | $[k,\ i,\ j]$ | $(0,0,0)$ |
| E | − | `nfy[k,i,j] · b[k+1,i,j]` | $Q_b^{y-}$ | $[k+1,\ i,\ j]$ | $(+1,0,0)$ |
| W | + | `nfy[k,i-1,j] · a[k,i-1,j]` | $Q_a^{y0}$ | $[k,\ i-1,\ j]$ | $(0,-1,0)$ |
| W | − | `nfy[k,i-1,j] · b[k,i-1,j]` | $Q_b^{y0}$ | $[k,\ i-1,\ j]$ | $(0,-1,0)$ |
| W | + | `nfy[k,i-1,j] · b[k+1,i-1,j]` | $Q_b^{y-}$ | $[k+1,\ i-1,\ j]$ | $(+1,-1,0)$ |
| N | + | `nfx[k,i,j] · a[k,i,j+1]` | $Q_a^{x-}$ | $[k,\ i,\ j+1]$ | $(0,0,+1)$ |
| N | + | `nfx[k,i,j] · c[k,i,j]` | $Q_c^{x0}$ | $[k,\ i,\ j]$ | $(0,0,0)$ |
| N | − | `nfx[k,i,j] · c[k+1,i,j]` | $Q_c^{x-}$ | $[k+1,\ i,\ j]$ | $(+1,0,0)$ |
| S | + | `nfx[k,i,j-1] · a[k,i,j-1]` | $Q_a^{x0}$ | $[k,\ i,\ j-1]$ | $(0,0,-1)$ |
| S | − | `nfx[k,i,j-1] · c[k,i,j-1]` | $Q_c^{x0}$ | $[k,\ i,\ j-1]$ | $(0,0,-1)$ |
| S | + | `nfx[k,i,j-1] · c[k+1,i,j-1]` | $Q_c^{x-}$ | $[k+1,\ i,\ j-1]$ | $(+1,0,-1)$ |

### Update (every frame)

```
E = Qa_y-[k,i+1,j]   + Qb_y0[k,i,j]     − Qb_y-[k+1,i,j]
W = Qa_y0[k,i-1,j]   − Qb_y0[k,i-1,j]   + Qb_y-[k+1,i-1,j]
N = Qa_x-[k,i,j+1]   + Qc_x0[k,i,j]     − Qc_x-[k+1,i,j]
S = Qa_x0[k,i,j-1]   − Qc_x0[k,i,j-1]   + Qc_x-[k+1,i,j-1]
D = nfy[k,i,j] + nfy[k,i-1,j] + nfx[k,i,j] + nfx[k,i,j-1]

a_new[k,i,j] = β·(E + W + N + S + J[k,i,j]) / D  +  (1 − β)·a[k,i,j]
```

### Local blocks vs. the physical blocks of Section 3

| Local block | z-frame | y-frame | x-frame |
|---|---|---|---|
| $Q_a^{y0}$, $Q_a^{y-}$ | $Q[z;y,0]$, $Q[z;y,-]$ | $Q[y;x,0]$, $Q[y;x,-]$ | $Q[x;z,0]$, $Q[x;z,-]$ |
| $Q_a^{x0}$, $Q_a^{x-}$ | $Q[z;x,0]$, $Q[z;x,-]$ | $Q[y;z,0]$, $Q[y;z,-]$ | $Q[x;y,0]$, $Q[x;y,-]$ |
| $Q_b^{y0}$, $Q_b^{y-}$ | $Q[x;y,0]$, $Q[x;y,-]$ | $Q[z;x,0]$, $Q[z;x,-]$ | $Q[y;z,0]$, $Q[y;z,-]$ |
| $Q_c^{x0}$, $Q_c^{x-}$ | $Q[y;x,0]$, $Q[y;x,-]$ | $Q[x;z,0]$, $Q[x;z,-]$ | $Q[z;y,0]$, $Q[z;y,-]$ |

**Why this matters for the circuit.** After $\pi^s$, the frame-local groups $(k, i, j)$ sit on the **same wires** in every frame. So the translations above are always the same incrementers on the same qubit groups:
- $\pm 1$ on the $i$-group;
- $\pm 1$ on the $j$-group;
- $+1$ on the $k$-group.

One z-frame circuit with fixed shifts serves all three frames.

**Check.** This table, applied in the z-, y- and x-frame layouts, reproduces the classical material solver with errors of $8.9\cdot10^{-16}$, $3.6\cdot10^{-15}$ and $1.8\cdot10^{-15}$ (8³ grid, finite-height L-block, $\mu_r = 100$).

---

## 6. Circuit procedure

**Option A: encode $Q$ directly.** Prepare the 12 blocks and $J$ as the state, then apply the shifts. This works, but computing $F\cdot A$ classically every step costs $O(N)$, which defeats a coherent algorithm.

**Option B (recommended): build $Q$ inside the circuit from $|A\rangle$.**

```
|A, J⟩ ⊗ |0⟩_slot(2 qubits: family c, slot s)
  1. H ⊗ H on the slot register                → 4 face slots per edge
  2. O_H: multiply by F_c[face(p, c, s)]        → amplitudes Q[d; c, s][p]
     (diagonal in (edge, slot): face index = p or p − e_t, computed from p)
  3. term register (Hadamards) + controlled translations of the index register by −δ
     (one ±1 incrementer/decrementer per non-zero component of δ, on a q-qubit group)
  4. signs ±1 as controlled phases (Z gates)
  5. interference / summation (Hadamards)       → E + W + N + S at index p
```

**Requirements and caveats.**
- **Per-axis padding** of every block to $(n,n,n)$, $n = 2^q$, so that a translation along $e_x$, $e_y$ or $e_z$ is an incrementer on one q-qubit group, with no carries between axes.
- **Wrap-around.** A q-qubit incrementer is cyclic: index $n-1$ wraps to $0$. Wrapped values land on boundary edges, which are not updated (Dirichlet), so discard them. The zero padding plane prevents contamination of interior terms.
- **Missing faces** at the domain boundary (e.g. $F_x[p-e_y]$ at $y=0$) give $Q = 0$, which is correct.
- **Frames by π.** The y- and x-frame tables are the z-frame table with the cyclic relabelling, so in a circuit one z-frame implementation plus the controlled rotations π, π² covers all three. π maps the block label $(d, c)$ to $(d{+}1, c{+}1)$ and rotates the index groups. See `test_cyclic_packing.py` and `test_cyclic_material.py`.

---

## 7. Verification

| File | What it checks |
|---|---|
| `test_edge_slot_products.py` | The two shift maps $F_z[p]A_x[p+e_y]$ and $F_z[p-e_y]A_x[p-e_y]$ (exact). The "crossed" shift is wrong (error 2.9). A full material Jacobi step from shifted $Q$ blocks equals the classical solver (error ~$10^{-14}$) |
| Table generator (this note) | Every one of the 36 table entries: the term equals block $Q[d;c,s]$ read at $p+\delta$, exactly, for all indices where it is defined (8³ grid, finite-height L-block, $\mu_r = 100$) |
| `test_cyclic_material.py` | The material update via π rotations with $F_x, F_y, F_z$ equals the classical solver |
| `test_face_edge_products.py` | The face-organised version (12 product vectors per face) equals the classical solver |

The test grid is $8^3$ with an L-shaped block of finite height, so $\nu$ has no rotation symmetry and every frame is tested non-trivially.
