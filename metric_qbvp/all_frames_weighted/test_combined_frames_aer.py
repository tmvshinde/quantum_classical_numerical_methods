"""
All three field components in ONE quantum circuit: frame register + controlled cyclic rotation.

Per Jacobi step the three components (A_z, A_y, A_x) are normally done by three circuit runs, one per
frame. Here a 2-qubit frame register |f>, f in {z=0, y=1, x=2}, puts all three in superposition:

    |Psi_0> = (1/N) sum_f |f> sum_tau |tau> |Q^(f)_{b(tau)}>        (blocks stored in the common (z,x,y) layout)

    1. controlled pi^f   : rotate the three index qubit groups (controlled-SWAP network) so that each
                           branch is in its own frame layout (k,i,j)
    2. SIGN              : CZ / CCZ on the term register
    3. STAR SHIFTER^dag  : one shifter for all three frames (frame-local offsets are identical)
    4. PREP^dag          : sums the 12 terms onto |term = 0>
    5. controlled pi^-f  : back to the common (z,x,y) layout
    read  <f, term=0, ancillas=0, p | Psi> * N * sqrt(12)  =  (E+W+N+S) of component f at edge p

Checks: one step with random fields (all three frames) and a full L-shape solve, against the
classical block solver. Run time and circuit runs are compared with the three-circuit version.
"""
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from star_shifter import build_proposed, TERMS

q = 3
n = nx = ny = nz = 2**q
L, mu0, mu_r, beta = 1.0, 1.0, 10.0, 0.8
h = L/(n - 1)

zxy_to_yzx = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1))
zxy_to_xyz = lambda a: np.ascontiguousarray(a.transpose(1, 2, 0))
xyz_to_zxy = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1))
yzx_to_zxy = lambda a: np.ascontiguousarray(a.transpose(1, 2, 0))
to_x = lambda a: np.ascontiguousarray(a)
to_y = lambda a: np.ascontiguousarray(a.transpose(1, 2, 0))
to_z = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1))
# frame layout -> storage layout (z,x,y), for (n,n,n) padded arrays
frame_to_storage = {'z': lambda a: a, 'y': yzx_to_zxy, 'x': xyz_to_zxy}
storage_to_frame = {'z': lambda a: a, 'y': zxy_to_yzx, 'x': zxy_to_xyz}
FRAMES = ['z', 'y', 'x']                     # frame register value f = 0, 1, 2

jx = np.zeros((nx-1, ny, nz)); jy = np.zeros((ny-1, nz, nx)); jz = np.zeros((nz-1, nx, ny))

# ---------------------------------------------------------------------------------------------
# material, face families, blocks, classical star sum (as in L_shape_mu_3d_blocks / _quantum_shift)
xc = (np.arange(n-1) + 0.5)*h
X, Y, Z = np.meshgrid(xc, xc, xc, indexing='ij')
L_mask = (((X >= 0.25) & (X <= 0.75) & (Y >= 0.25) & (Y <= 0.50)) |
          ((X >= 0.50) & (X <= 0.75) & (Y >= 0.50) & (Y <= 0.75)))
nu_cell = 1.0/np.where(L_mask, mu_r*mu0, mu0)

def face_family(nu, axis):
    nu = np.moveaxis(nu, axis, 0)
    f = np.empty((nu.shape[0] + 1,) + nu.shape[1:])
    f[1:-1] = 0.5*(nu[:-1] + nu[1:]); f[0], f[-1] = nu[0], nu[-1]
    return np.moveaxis(f, 0, axis)

Fx, Fy, Fz = (face_family(nu_cell, ax) for ax in (0, 1, 2))
faces = {'z': (to_z(Fy), to_z(Fx)), 'y': (to_y(Fx), to_y(Fz)), 'x': (to_x(Fz), to_x(Fy))}

BLOCK_NAMES = ['Qa_y0', 'Qa_ym', 'Qa_x0', 'Qa_xm', 'Qb_y0', 'Qb_ym', 'Qc_x0', 'Qc_xm']
def blocks(a, b, c, nfy, nfx):
    Qa_y0 = np.zeros_like(a); Qa_y0[:, :-1, :] = nfy * a[:, :-1, :]
    Qa_ym = np.zeros_like(a); Qa_ym[:, 1:, :]  = nfy * a[:, 1:, :]
    Qa_x0 = np.zeros_like(a); Qa_x0[:, :, :-1] = nfx * a[:, :, :-1]
    Qa_xm = np.zeros_like(a); Qa_xm[:, :, 1:]  = nfx * a[:, :, 1:]
    Qb_y0 = np.zeros_like(b); Qb_y0[:-1, :, :] = nfy * b[:-1, :, :]
    Qb_ym = np.zeros_like(b); Qb_ym[1:, :, :]  = nfy * b[1:, :, :]
    Qc_x0 = np.zeros_like(c); Qc_x0[:-1, :, :] = nfx * c[:-1, :, :]
    Qc_xm = np.zeros_like(c); Qc_xm[1:, :, :]  = nfx * c[1:, :, :]
    return dict(zip(BLOCK_NAMES, [Qa_y0, Qa_ym, Qa_x0, Qa_xm, Qb_y0, Qb_ym, Qc_x0, Qc_xm]))

K, K1 = slice(0, n-1), slice(1, n)
I, Im, Ip = slice(1, n-1), slice(0, n-2), slice(2, n)
J_, Jm, Jp = slice(1, n-1), slice(0, n-2), slice(2, n)

def star_sum_classical(a, b, c, nfy, nfx):
    Q = blocks(a, b, c, nfy, nfx)
    E = Q['Qa_ym'][K, Ip, J_] + Q['Qb_y0'][K, I, J_]  - Q['Qb_ym'][K1, I, J_]
    W = Q['Qa_y0'][K, Im, J_] - Q['Qb_y0'][K, Im, J_] + Q['Qb_ym'][K1, Im, J_]
    N = Q['Qa_xm'][K, I, Jp]  + Q['Qc_x0'][K, I, J_]  - Q['Qc_xm'][K1, I, J_]
    S = Q['Qa_x0'][K, I, Jm]  - Q['Qc_x0'][K, I, Jm]  + Q['Qc_xm'][K1, I, Jm]
    return E + W + N + S

def diag_D(nfy, nfx):
    return nfy[K, I, J_] + nfy[K, Im, J_] + nfx[K, I, J_] + nfx[K, I, Jm]

def frames_of(ax, ay, az):
    return {'z': (az, ax, ay),
            'y': (zxy_to_yzx(ay), zxy_to_yzx(az), zxy_to_yzx(ax)),
            'x': (zxy_to_xyz(ax), zxy_to_xyz(ay), zxy_to_xyz(az))}

# ---------------------------------------------------------------------------------------------
# quantum circuit
TERM_BLOCK = {('E', 'a'): 'Qa_ym', ('E', 'bc_k'): 'Qb_y0', ('E', 'bc_k1'): 'Qb_ym',
              ('W', 'a'): 'Qa_y0', ('W', 'bc_k'): 'Qb_y0', ('W', 'bc_k1'): 'Qb_ym',
              ('N', 'a'): 'Qa_xm', ('N', 'bc_k'): 'Qc_x0', ('N', 'bc_k1'): 'Qc_xm',
              ('S', 'a'): 'Qa_x0', ('S', 'bc_k'): 'Qc_x0', ('S', 'bc_k1'): 'Qc_xm'}
term_code = lambda al, sg, r1, r0: al + 2*sg + 4*r1 + 8*r0

shifter, cnt, regs = build_proposed(q)
NQS = shifter.num_qubits                     # 4q + 6 qubits of the shifter
G = [list(range(0, q)), list(range(q, 2*q)), list(range(2*q, 3*q))]   # index groups (LSB group first)
tq = [3*q, 3*q + 1, 3*q + 2, 3*q + 3]        # term register (alpha, sigma, r1, r0)
FQ = [NQS, NQS + 1]                          # frame register: f = 0 (z), 1 (y, bit 0), 2 (x, bit 1)
NQ = NQS + 2

def sign_circuit():
    c = QuantumCircuit(4)
    c.x(1); c.cz(1, 2); c.x(1)
    c.x([2, 3]); c.ccz(1, 2, 3); c.x([2, 3])
    return c

def prep_circuit():
    c = QuantumCircuit(4)
    c.h(0); c.h(1)
    c.ry(2*np.arccos(np.sqrt(2/3)), 2)
    c.x(2); c.ch(2, 3); c.x(2)
    return c

def cswap_groups(qc, ctrl, ga, gb):
    for a, b in zip(ga, gb):
        qc.cswap(ctrl, a, b)

def controlled_pi(qc, inverse=False):
    """storage (z,x,y) -> frame layout, controlled by the frame register.
    y-frame (f bit 0): (z,x,y) -> (y,z,x) = swap(G0,G2), swap(G1,G2)
    x-frame (f bit 1): (z,x,y) -> (x,y,z) = swap(G0,G1), swap(G1,G2)"""
    steps = [(FQ[0], G[0], G[2]), (FQ[0], G[1], G[2]),
             (FQ[1], G[0], G[1]), (FQ[1], G[1], G[2])]
    y_steps, x_steps = steps[:2], steps[2:]
    if inverse:
        y_steps, x_steps = y_steps[::-1], x_steps[::-1]
    for st in y_steps + x_steps:
        cswap_groups(qc, *st)

body = QuantumCircuit(NQ)
controlled_pi(body)                                                   # 1.
body.compose(sign_circuit(), qubits=tq, inplace=True)                 # 2.
body.compose(shifter.inverse(), qubits=list(range(NQS)), inplace=True)  # 3.
body.compose(prep_circuit().inverse(), qubits=tq, inplace=True)       # 4.
controlled_pi(body, inverse=True)                                     # 5.
sim = AerSimulator(method='statevector')
body_t = transpile(body, sim, optimization_level=0)   # level 0: body acts on an encoded state

def pad(arr):
    out = np.zeros((n, n, n)); out[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr
    return out

to_flat = lambda arr: arr.transpose(2, 1, 0).ravel()          # [g0, g1, g2] -> g0 + n g1 + n^2 g2
from_flat = lambda v: v.reshape(n, n, n).transpose(2, 1, 0)

def quantum_star_sum_all(ax, ay, az):
    """E+W+N+S for all three components from ONE circuit run. Returns {frame: interior array}."""
    fr = frames_of(ax, ay, az)
    amp = np.zeros((3, 16, n**3))
    for f_idx, f in enumerate(FRAMES):
        Q = blocks(*fr[f], *faces[f])
        for (face, mem, al, sg, r1, r0) in TERMS:
            block_storage = frame_to_storage[f](pad(Q[TERM_BLOCK[(face, mem)]]))   # common layout
            amp[f_idx, term_code(al, sg, r1, r0)] = to_flat(block_storage)
    norm = np.linalg.norm(amp)
    psi0 = np.zeros(2**NQ, dtype=complex)
    for f_idx in range(3):
        base = f_idx*2**NQS                     # frame register on the two top qubits
        psi0[base: base + 16*n**3] = amp[f_idx].ravel()/norm

    qc = QuantumCircuit(NQ)
    qc.set_statevector(psi0)
    qc.compose(body_t, inplace=True)
    qc.save_statevector()
    sv = sim.run(qc).result().get_statevector().data

    out = {}
    for f_idx, f in enumerate(FRAMES):
        base = f_idx*2**NQS
        res_storage = from_flat(sv[base: base + n**3])          # term=0, ancillas=0, storage layout
        res_frame = storage_to_frame[f](res_storage)*norm*np.sqrt(12)
        out[f] = res_frame[K, I, J_]
    return out, sv

# ---------------------------------------------------------------------------------------------
def relax_from_sum(a, J, nfy, nfx, S):
    a_new = np.zeros_like(a)
    a_new[K, I, J_] = beta*(S + J[K, I, J_])/diag_D(nfy, nfx) + (1 - beta)*a[K, I, J_]
    return a_new

def update_quantum_combined(ax, ay, az):
    S, _ = quantum_star_sum_all(ax, ay, az)
    fr = frames_of(ax, ay, az); Js = {'z': jz, 'y': jy, 'x': jx}
    new = {f: relax_from_sum(fr[f][0], Js[f], *faces[f], S[f].real) for f in FRAMES}
    return xyz_to_zxy(new['x']), yzx_to_zxy(new['y']), new['z']

def update_classical(ax, ay, az):
    fr = frames_of(ax, ay, az); Js = {'z': jz, 'y': jy, 'x': jx}
    new = {f: relax_from_sum(fr[f][0], Js[f], *faces[f], star_sum_classical(*fr[f], *faces[f]))
           for f in FRAMES}
    return xyz_to_zxy(new['x']), yzx_to_zxy(new['y']), new['z']

def apply_bc_az(az, A_left=1e-2, A_right=5e-2):
    az[:, 0, :] = A_left; az[:, -1, :] = A_right
    az[:, 1:-1, 0] = az[:, 1:-1, 1]; az[:, 1:-1, -1] = az[:, 1:-1, -2]
    return az

def solve(update, timesteps=2000, conv_tol=1e-7):
    ax = np.zeros((nz, nx-1, ny)); ay = np.zeros((nz, nx, ny-1)); az = apply_bc_az(np.zeros((nz-1, nx, ny)))
    for t in range(timesteps):
        axn, ayn, azn = update(ax, ay, az)
        azn = apply_bc_az(azn)
        err = np.linalg.norm(azn - az)**2
        ax, ay, az = axn, ayn, azn
        if err < conv_tol:
            break
    return ax, ay, az, t


if __name__ == "__main__":
    ops = dict(body.count_ops())
    print(f"combined circuit: {NQ} qubits ({NQS} shifter + 2 frame), gates: {ops}")
    print(f"  controlled pi forward + back: {ops.get('cswap', 0)} Fredkin (controlled-SWAP) gates\n")

    # one step, random fields, all three frames from one run
    rng = np.random.default_rng(0)
    tax = rng.standard_normal((nz, nx-1, ny)); tay = rng.standard_normal((nz, nx, ny-1))
    taz = rng.standard_normal((nz-1, nx, ny))
    Sq, sv = quantum_star_sum_all(tax, tay, taz)
    fr = frames_of(tax, tay, taz)
    print("one step, random fields (one circuit run for all three components):")
    ok = True
    for f in FRAMES:
        Sc = star_sum_classical(*fr[f], *faces[f])
        err = np.abs(Sq[f] - Sc).max()
        ok &= err < 1e-12
        print(f"  {f}-frame: max|E+W+N+S quantum - classical| = {err:.1e},  max|imag| = {np.abs(Sq[f].imag).max():.1e}")
    p_ok = sum(np.sum(np.abs(sv[f*2**NQS: f*2**NQS + n**3])**2) for f in range(3))
    print(f"  P(term=0, ancillas=0), summed over frames = {p_ok:.4f}\n")

    # full solve
    t0 = time.time(); axq, ayq, azq, tq_ = solve(update_quantum_combined); t_q = time.time() - t0
    axc, ayc, azc, tc_ = solve(update_classical)
    print("full L-shape solve:")
    print(f"  combined quantum circuit: converged at step {tq_}  ({t_q:.1f} s, {tq_ + 1} circuit runs)")
    print(f"  classical blocks        : converged at step {tc_}")
    print(f"  max |az_quantum - az_classical| = {np.abs(azq - azc).max():.1e}")
    ok &= np.abs(azq - azc).max() < 1e-10 and tq_ == tc_
    np.savetxt("results/azq_combined.txt",
               np.column_stack([*(g.ravel() for g in np.meshgrid(*(np.arange(s) for s in azq.shape), indexing='ij')),
                                azq.ravel()]),
               fmt=['%d', '%d', '%d', '%.16e'], header="A_z, combined three-frame circuit; columns: k i j Az")
    print("  saved results/azq_combined.txt")
    print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED")
