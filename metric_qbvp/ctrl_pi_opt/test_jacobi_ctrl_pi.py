"""
Jacobi step for all three frames from ONE encoding in the storage layout, using controlled-pi.

    a_new_f = beta * ( S_f - J_f ) / D_f  +  (1 - beta) * a_f            (f = z, y, x)

Encoded ONCE, in the common storage layout (z, x, y), with a block label L:
    L = 0..11  : the 12 edge-face products  A_d * F_c (each edge component times its 4 incident faces)
    L = 12..14 : J_z, J_y, J_x
    L = 15..17 : a_d * D_d   (retained term; D_d is multiplied in so that the common 1/D below cancels)

Circuit (q = 2, 4^3 grid, 24 qubits):
    FRAME   uniform superposition over f in {0: z, 1: y, 2: x}           (Ry + CH, exact)
    COPY    H on 1 qubit (a product block used by two terms in a frame gets two branches)
    ctrl-pi controlled SWAPs: storage layout -> frame layout in each frame branch
    MAP     (L, f, copy) -> term code (MCX lookup, constant size)
    STAR SHIFTER^dag   all 12 shifts, all frames
    UNMAP
    W1      ancilla w1: 1/D_f(p) at the TARGET edge p (lookup over index and frame)
    W2      ancilla w2: sign*beta (star), +beta (J), 1-beta (a), 0 (unused)  (lookup over L, f, copy)
    SUM     H on L and copy (not on frame)
    read    <f, rest = 0, p|Psi> * 8 * sqrt(6) * N * max(1/D)  =  a_new_f(p)   (frame layout)
"""
import os, time
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import UCRYGate
from qiskit_aer import AerSimulator
from star_shifter import build_proposed, TERMS, expected_offset
from test_combined_frames_aer import blocks, TERM_BLOCK

q = 2
n = 2**q
beta = 0.8
FRAMES = ['z', 'y', 'x']

# ---------------------------------------------------------------- layouts (storage = (z, x, y))
zxy_to_yzx = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1))
zxy_to_xyz = lambda a: np.ascontiguousarray(a.transpose(1, 2, 0))
xyz_to_zxy = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1))
yzx_to_zxy = lambda a: np.ascontiguousarray(a.transpose(1, 2, 0))
to_x = lambda a: np.ascontiguousarray(a)
to_y = lambda a: np.ascontiguousarray(a.transpose(1, 2, 0))
to_z = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1))
storage_to_frame = {'z': lambda a: a, 'y': zxy_to_yzx, 'x': zxy_to_xyz}
frame_to_storage = {'z': lambda a: a, 'y': yzx_to_zxy, 'x': xyz_to_zxy}
pad = lambda x: np.pad(x, [(0, n - s) for s in x.shape])
to_flat = lambda x: x.transpose(2, 1, 0).ravel()                     # [k, i, j] -> k + n i + n^2 j
from_flat = lambda v: v.reshape(n, n, n).transpose(2, 1, 0)
K, K1 = slice(0, n-1), slice(1, n)
I, Im, Ip = slice(1, n-1), slice(0, n-2), slice(2, n)
J_, Jm, Jp = slice(1, n-1), slice(0, n-2), slice(2, n)


def face_family(nu, axis):
    nu = np.moveaxis(nu, axis, 0); f = np.empty((nu.shape[0] + 1,) + nu.shape[1:])
    f[1:-1] = 0.5*(nu[:-1] + nu[1:]); f[0], f[-1] = nu[0], nu[-1]
    return np.moveaxis(f, 0, axis)


def frames_of(ax, ay, az):
    return {'z': (az, ax, ay),
            'y': (zxy_to_yzx(ay), zxy_to_yzx(az), zxy_to_yzx(ax)),
            'x': (zxy_to_xyz(ax), zxy_to_xyz(ay), zxy_to_xyz(az))}


def star_sum(a, b, c, nfy, nfx):
    Q = blocks(a, b, c, nfy, nfx)
    return (Q['Qa_ym'][K, Ip, J_] + Q['Qb_y0'][K, I, J_] - Q['Qb_ym'][K1, I, J_]
            + Q['Qa_y0'][K, Im, J_] - Q['Qb_y0'][K, Im, J_] + Q['Qb_ym'][K1, Im, J_]
            + Q['Qa_xm'][K, I, Jp] + Q['Qc_x0'][K, I, J_] - Q['Qc_xm'][K1, I, J_]
            + Q['Qa_x0'][K, I, Jm] - Q['Qc_x0'][K, I, Jm] + Q['Qc_xm'][K1, I, Jm])


def diag_D(nfy, nfx):
    return nfy[K, I, J_] + nfy[K, Im, J_] + nfx[K, I, J_] + nfx[K, I, Jm]


NEG = {('E', 'bc_k1'), ('W', 'bc_k'), ('N', 'bc_k1'), ('S', 'bc_k')}

# ---------------------------------------------------------------- qubits
shifter, _, _ = build_proposed(q)
NQS = shifter.num_qubits                                  # 4q + 6 = 14
IDX = list(range(3*q)); TQ = [3*q + t for t in range(4)]
G = [list(range(0, q)), list(range(q, 2*q)), list(range(2*q, 3*q))]
LQ = list(range(NQS, NQS + 5)); FQ = [NQS + 5, NQS + 6]; CQ = NQS + 7
W1, W2 = NQS + 8, NQS + 9
NQ = NQS + 10                                             # 24
sim = AerSimulator(method="statevector")


# ---------------------------------------------------------------- classical data -> encoding + lookups
def problem(ax, ay, az, Js, faces):
    """encoded vector (storage layout), MAP table, W1/W2 angles, normalisation"""
    fr = frames_of(ax, ay, az)
    # 1. the 12 physical products: every frame's 8 product blocks mapped back to storage, deduplicated
    phys, label_of = [], {}
    for f in FRAMES:
        Q = blocks(*fr[f], *faces[f])
        for nm in ['Qa_y0', 'Qa_ym', 'Qa_x0', 'Qa_xm', 'Qb_y0', 'Qb_ym', 'Qc_x0', 'Qc_xm']:
            s = frame_to_storage[f](pad(Q[nm]))
            hit = [L for L, p in enumerate(phys) if np.allclose(p, s)]
            if not hit:
                phys.append(s); hit = [len(phys) - 1]
            label_of[(f, nm)] = hit[0]
    assert len(phys) == 12, f"expected 12 frame-independent products, got {len(phys)}"

    # 2. J and a*D per frame (storage layout), and 1/D per frame (frame layout)
    invD, blocks_all = {}, list(phys)
    Dfull = {}
    for f in FRAMES:
        D = np.zeros((n, n, n)); D[K, I, J_] = diag_D(*faces[f]); Dfull[f] = D
        invD[f] = np.where(D > 0, 1/np.where(D > 0, D, 1), 0.0)
    for f in FRAMES:
        blocks_all.append(frame_to_storage[f](pad(Js[f])))
    for f in FRAMES:
        blocks_all.append(frame_to_storage[f](pad(fr[f][0])*Dfull[f]))
    enc = np.zeros((32, n**3))
    for L, blk in enumerate(blocks_all):
        enc[L] = to_flat(blk)
    norm = np.linalg.norm(enc)

    # 3. MAP table and constants: (L, f, copy) -> term code, constant
    table = {}
    for fi, f in enumerate(FRAMES):
        used = {}
        for (face, mem, al, sg, r1, r0) in TERMS:
            L = label_of[(f, TERM_BLOCK[(face, mem)])]
            cp = used.get(L, 0); used[L] = cp + 1
            table[(L, fi, cp)] = ((al, sg, r1, r0), (-beta if (face, mem) in NEG else beta))
        table[(12 + fi, fi, 0)] = ((0, 0, 0, 0), beta)             # +beta J  (paper: S + J)
        table[(15 + fi, fi, 0)] = ((0, 0, 0, 0), 1 - beta)         # (1 - beta) a
    const = np.zeros(32*4*2)
    for (L, fi, cp), (_, cval) in table.items():
        const[L + 32*fi + 128*cp] = cval
    w2_angles = 2*np.arccos(const)                                  # unused entries: const 0
    invD_max = max(v.max() for v in invD.values())
    w1_vals = np.zeros(n**3*4)
    for fi, f in enumerate(FRAMES):
        w1_vals[fi*n**3:(fi + 1)*n**3] = to_flat(invD[f])/invD_max
    w1_angles = 2*np.arccos(w1_vals)
    return enc.ravel()/norm, norm, table, w1_angles, w2_angles, invD_max


# ---------------------------------------------------------------- circuit
def frame_prep(c, fq):                     # (|0> + |1> + |2>)/sqrt(3) on f = f0 + 2 f1
    c.ry(2*np.arccos(np.sqrt(2/3)), fq[1]); c.x(fq[1]); c.ch(fq[1], fq[0]); c.x(fq[1])


def ctrl_pi(c):                            # storage (z,x,y) -> frame layout, controlled by the frame bits
    for ctrl, ga, gb in [(FQ[0], G[0], G[2]), (FQ[0], G[1], G[2]), (FQ[1], G[0], G[1]), (FQ[1], G[1], G[2])]:
        for a_, b_ in zip(ga, gb):
            c.cswap(ctrl, a_, b_)


def lookup(c, table):
    for (L, fi, cp), (code, _) in table.items():
        for bit, v in zip(TQ, code):
            if v:
                c.mcx(LQ + FQ + [CQ], bit, ctrl_state=L + 32*fi + 128*cp)


def build_body(table, w1_angles, w2_angles, regs=None, barriers=False):
    b = QuantumCircuit(*regs) if regs else QuantumCircuit(NQ)
    bar = (lambda lab: b.barrier(label=lab)) if barriers else (lambda lab: None)
    bar("FRAME, COPY"); frame_prep(b, FQ); b.h(CQ)
    bar("ctrl-π"); ctrl_pi(b)
    bar("MAP"); lookup(b, table)
    bar("STAR SHIFTER†"); b.compose(shifter.inverse(), qubits=list(range(NQS)), inplace=True)
    bar("UNMAP"); lookup(b, table)
    bar("W1: 1/D(p)"); b.append(UCRYGate(list(w1_angles)), [W1] + IDX + FQ)
    bar("W2: β, 1−β, signs"); b.append(UCRYGate(list(w2_angles)), [W2] + LQ + FQ + [CQ])
    bar("SUM"); b.h(LQ + [CQ])
    return b


def jacobi_step_quantum(ax, ay, az, Js, faces):
    vec, norm, table, w1a, w2a, invD_max = problem(ax, ay, az, Js, faces)
    # index at qubits 0..3q-1, label L at qubits LQ: flat index p + 2^LQ[0] * L (term, flags, carries 0)
    psi0 = np.zeros(2**NQ, complex)
    for L in range(32):
        psi0[L*2**LQ[0]: L*2**LQ[0] + n**3] = vec[L*n**3:(L + 1)*n**3]
    body = transpile(build_body(table, w1a, w2a), sim, optimization_level=0)
    qc = QuantumCircuit(NQ); qc.set_statevector(psi0); qc.compose(body, inplace=True); qc.save_statevector()
    sv = sim.run(qc).result().get_statevector().data
    out, p = {}, 0.0
    for fi, f in enumerate(FRAMES):
        ro = sv[fi*2**FQ[0]: fi*2**FQ[0] + n**3]
        p += np.sum(np.abs(ro)**2)
        full = np.zeros((n-1, n, n)); full[K, I, J_] = (from_flat(ro).real*8*np.sqrt(6)*norm*invD_max)[K, I, J_]
        out[f] = full                                           # frame layout, interior updated
    return out, p, (table, w1a, w2a)


def jacobi_step_classical(ax, ay, az, Js, faces):
    fr = frames_of(ax, ay, az); out = {}
    for f in FRAMES:
        a, b, c = fr[f]; nfy, nfx = faces[f]
        full = np.zeros((n-1, n, n))
        full[K, I, J_] = beta*(star_sum(a, b, c, nfy, nfx) + Js[f][K, I, J_])/diag_D(nfy, nfx) \
                         + (1 - beta)*a[K, I, J_]
        out[f] = full
    return out


def to_storage(out):
    """frame-layout results -> ax, ay, az in the storage layout (z, x, y), native shapes"""
    return (xyz_to_zxy(out['x'])[:, :n-1, :],                # ax: (n, n-1, n)
            yzx_to_zxy(out['y'])[:, :, :n-1],                # ay: (n, n, n-1)
            out['z'])                                        # az: (n-1, n, n)


def save_field_txt(path, arr, title):
    """3-D array as text, slice by slice along the first axis (k = z), keeping its shape."""
    with open(path, "w") as fh:
        fh.write(f"# {title}\n# shape {arr.shape} = (z, x, y) storage layout; one block per k, rows = x, columns = y\n")
        for k in range(arr.shape[0]):
            fh.write(f"\n# k = {k}\n")
            np.savetxt(fh, arr[k], fmt="%+.16e")


def named_circuits(table, w1a, w2a):
    regs = list(shifter.qregs) + [QuantumRegister(5, "L"), QuantumRegister(2, "frame"),
                                  QuantumRegister(1, "copy"), QuantumRegister(1, "w1"), QuantumRegister(1, "w2")]
    enc = IDX + LQ
    det = QuantumCircuit(*regs, name="jacobi_ctrl_pi_gates")
    det.append(Gate("ENCODE products, J, a·D", len(enc), []), enc)
    det.compose(build_body(table, w1a, w2a, regs=regs, barriers=True), inplace=True)
    blk = QuantumCircuit(*regs, name="jacobi_ctrl_pi_blocks")
    blk.append(Gate("ENCODE products, J, a·D", len(enc), []), enc)
    stages, cur, label = [], [], None
    for ins in det.data[1:]:
        if ins.operation.name == "barrier":
            if cur: stages.append((label, cur))
            label, cur = ins.operation.label, []
        else:
            cur.append(ins)
    stages.append((label, cur))
    for label, ins_list in stages:
        qs = sorted({det.find_bit(x).index for ins in ins_list for x in ins.qubits})
        sub = QuantumCircuit(len(qs))
        for ins in ins_list:
            sub.append(ins.operation, [qs.index(det.find_bit(x).index) for x in ins.qubits])
        blk.append(sub.to_gate(label=label), qs)
    return {"jacobi_ctrl_pi_blocks": (blk, {"fold": -1, "scale": 0.8}),
            "jacobi_ctrl_pi_gates": (det, {"fold": 60, "scale": 0.6})}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    nu_cell = rng.uniform(0.05, 1.0, (n-1, n-1, n-1))                 # random material in every cell
    Fx, Fy, Fz = (face_family(nu_cell, ax_) for ax_ in (0, 1, 2))
    faces = {'z': (to_z(Fy), to_z(Fx)), 'y': (to_y(Fx), to_y(Fz)), 'x': (to_x(Fz), to_x(Fy))}
    ax = rng.standard_normal((n, n-1, n)); ay = rng.standard_normal((n, n, n-1)); az = rng.standard_normal((n-1, n, n))
    Js = {f: rng.standard_normal((n-1, n, n)) for f in FRAMES}

    print(f"{NQ} qubits (q = {q}, grid {n}^3, random nu per cell), beta = {beta}", flush=True)
    t0 = time.time()
    out_q, p, (table, w1a, w2a) = jacobi_step_quantum(ax, ay, az, Js, faces)
    print(f"one circuit run for all three frames: {time.time()-t0:.1f} s, P(success) = {p:.2e}", flush=True)
    out_c = jacobi_step_classical(ax, ay, az, Js, faces)
    for f in FRAMES:
        print(f"  {f}-frame: max|a_new quantum - classical| = {np.abs(out_q[f] - out_c[f]).max():.1e}", flush=True)

    # save ax, ay, az (new values) for both, in their native shapes
    res = "results/jacobi_ctrl_pi"; os.makedirs(res, exist_ok=True)
    for tag, out in (("quantum", out_q), ("classical", out_c)):
        axn, ayn, azn = to_storage(out)
        for name, arr in (("ax", axn), ("ay", ayn), ("az", azn)):
            save_field_txt(f"{res}/{name}_{tag}.txt", arr,
                           f"{name} after one Jacobi step ({tag}); interior updated, boundary edges 0")
    print(f"saved {res}/ax|ay|az_quantum.txt and ax|ay|az_classical.txt", flush=True)

    from circuit_saving import save_circuits
    save_circuits(named_circuits(table, w1a, w2a), "circuits/jacobi_ctrl_pi")
