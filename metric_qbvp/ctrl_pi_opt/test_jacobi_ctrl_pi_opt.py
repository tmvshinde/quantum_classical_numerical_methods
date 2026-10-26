"""
Optimised MAP / UNMAP / W2 for test_jacobi_ctrl_pi.py: a STRUCTURED block label.

Old: L = 0..17 (arbitrary numbering)  ->  MAP = 39 MCX with 8 controls, W2 = 256-entry lookup.
New: L = s + 2h + 4d + 16*kind   (still 5 qubits)
        kind 0 : product A_d * F   d = edge component (x=0, y=1, z=2)
                                   h = which of its two face normals (d+1+h mod 3)
                                   s = face slot (0: same index, 1: one step back)
        kind 1 : d = component; s = 0 -> J_d,  s = 1 -> a_d * D_d
The frame only changes the edge's ROLE r = d - k_f (mod 3)  (k_f = 2, 1, 0 for frames z, y, x);
h and s are frame independent. So
    MAP-A : d -> r     controlled mod-3 increment (frame z) / decrement (frame y) on the 2-bit d register
    MAP-B : term bits from (kind, r, h, s, copy): 8 small MCX (<= 4 controls), same for every frame
            a-terms (r=0):  alpha = !h, sigma = !s, r1 = 0, r0 = 1
            b-terms (r=1):  alpha = 0,  sigma = copy, r1 = s, r0 = 0     (h = 1: not used, weight 0)
            c-terms (r=2):  alpha = 1,  sigma = copy, r1 = s, r0 = 0     (h = 0: not used, weight 0)
    W2    : constants from (kind, r, h, s, copy): 64-entry lookup
            star: +beta (a-terms), (-1)^(s xor copy) beta (b, c terms);  J: +beta;  a*D: 1 - beta
Everything else (encoding layout, frame prep, ctrl-pi, star shifter, W1 = 1/D at the target, SUM)
is as in test_jacobi_ctrl_pi.py, which is only imported.
"""
import os, time
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import UCRYGate, UnitaryGate
import test_jacobi_ctrl_pi as T

n, q, beta, FRAMES = T.n, T.q, T.beta, T.FRAMES
LQ, FQ, CQ, TQ, IDX = T.LQ, T.FQ, T.CQ, T.TQ, T.IDX
S_, H_, D0, D1, KIND = LQ                      # label qubits: s, h, d (2 bits), kind
KF = {'z': 2, 'y': 1, 'x': 0}                  # physical axis of the frame's k direction (x=0, y=1, z=2)
ROLE = {'Qa_y0': (0, 2, 0), 'Qa_ym': (0, 2, 1), 'Qa_x0': (0, 1, 0), 'Qa_xm': (0, 1, 1),   # (r, local normal, s)
        'Qb_y0': (1, 2, 0), 'Qb_ym': (1, 2, 1), 'Qc_x0': (2, 1, 0), 'Qc_xm': (2, 1, 1)}


def label(kind, d, h, s):
    return s + 2*h + 4*d + 16*kind


def encode(ax, ay, az, Js, faces):
    """encoded vector with the structured labels; checks that the products are frame independent"""
    fr = T.frames_of(ax, ay, az)
    blocks_by_label = {}
    for f in FRAMES:
        Q = T.blocks(*fr[f], *faces[f])
        for nm, (r, normal, s) in ROLE.items():
            h = (normal - r - 1) % 3
            d = (r + KF[f]) % 3
            arr = T.frame_to_storage[f](T.pad(Q[nm]))
            L = label(0, d, h, s)
            if L in blocks_by_label:
                assert np.allclose(blocks_by_label[L], arr), "product not frame independent"
            blocks_by_label[L] = arr
    assert len(blocks_by_label) == 12
    Dfull, invD = {}, {}
    for f in FRAMES:
        D = np.zeros((n, n, n)); D[T.K, T.I, T.J_] = T.diag_D(*faces[f]); Dfull[f] = D
        invD[f] = np.where(D > 0, 1/np.where(D > 0, D, 1), 0.0)
        blocks_by_label[label(1, KF[f], 0, 0)] = T.frame_to_storage[f](T.pad(Js[f]))              # J_d
        blocks_by_label[label(1, KF[f], 0, 1)] = T.frame_to_storage[f](T.pad(fr[f][0])*Dfull[f])  # a_d D_d
    enc = np.zeros((32, n**3))
    for L, arr in blocks_by_label.items():
        enc[L] = T.to_flat(arr)
    norm = np.linalg.norm(enc)
    invD_max = max(v.max() for v in invD.values())
    w1 = np.zeros(4*n**3)
    for fi, f in enumerate(FRAMES):
        w1[fi*n**3:(fi + 1)*n**3] = T.to_flat(invD[f])/invD_max
    return enc/norm, norm, 2*np.arccos(w1), invD_max


# ---------------------------------------------------------------- structured MAP and W2
INC = np.zeros((4, 4))
for v_old, v_new in [(0, 1), (1, 2), (2, 0), (3, 3)]:
    INC[v_new, v_old] = 1                                   # mod-3 increment on a 2-bit register


def mod3_step(c, frame_state, up):
    """controlled +1 (up) or -1 mod 3 on the 2-bit register (D0, D1), phase free:
    +1 = swap(00,01) then swap(00,10);  -1 = the reverse order.
    swap(00,01): flip D0 if D1 = 0;   swap(00,10): flip D1 if D0 = 0   (plus the frame controls)"""
    s01 = lambda: c.mcx(FQ + [D1], D0, ctrl_state=frame_state)          # D1 = 0 (bit 2 of ctrl_state = 0)
    s02 = lambda: c.mcx(FQ + [D0], D1, ctrl_state=frame_state)          # D0 = 0
    if up:
        s01(); s02()
    else:
        s02(); s01()


def map_a(c, inverse=False):
    """d -> r = d - k_f (mod 3):  frame z (f=0): +1,  frame y (f=1): -1,  frame x (f=2): none"""
    mod3_step(c, 0b00, up=not inverse)          # frame z
    mod3_step(c, 0b01, up=inverse)              # frame y


def map_b(c):
    """term bits (alpha, sigma, r1, r0) from (kind, r = (D1 D0), h, s, copy); kind = 1 -> 0000"""
    al, sg, r1, r0 = TQ
    c.mcx([KIND, D0, D1], r0, ctrl_state=0b000)                          # r0 = a-term
    c.mcx([KIND, D0, S_], r1, ctrl_state=0b110)                          # r1 = b-term and s
    c.mcx([KIND, D1, S_], r1, ctrl_state=0b110)                          #    + c-term and s
    c.mcx([KIND, D0, D1, H_], al, ctrl_state=0b0000)                     # alpha = a-term and !h
    c.mcx([KIND, D1], al, ctrl_state=0b10)                               #       + c-term
    c.mcx([KIND, D0, D1, S_], sg, ctrl_state=0b0000)                     # sigma = a-term and !s
    c.mcx([KIND, D0, CQ], sg, ctrl_state=0b110)                          #       + b-term and copy
    c.mcx([KIND, D1, CQ], sg, ctrl_state=0b110)                          #       + c-term and copy


def w2_angles():
    """constants over (kind, r, h, s, copy); controls ordered [s, h, d0, d1, kind, copy]"""
    const = np.zeros(64)
    for x in range(64):
        s, h, r, kind, cp = x & 1, (x >> 1) & 1, (x >> 2) & 3, (x >> 4) & 1, (x >> 5) & 1
        if kind == 0:
            if r == 0 and cp == 0:
                const[x] = beta                                          # a-terms (all +)
            elif (r == 1 and h == 0) or (r == 2 and h == 1):
                const[x] = (-beta) if (s ^ cp) else beta                 # b-, c-terms
        elif r == 0 and h == 0 and cp == 0:
            const[x] = beta if s == 0 else (1 - beta)                    # +J  /  a*D
    return 2*np.arccos(const)


def build_body(w1a, regs=None, barriers=False):
    b = QuantumCircuit(*regs) if regs else QuantumCircuit(T.NQ)
    bar = (lambda lab: b.barrier(label=lab)) if barriers else (lambda lab: None)
    bar("FRAME, COPY"); T.frame_prep(b, FQ); b.h(CQ)
    bar("ctrl-π"); T.ctrl_pi(b)
    bar("MAP-A d→r"); map_a(b)
    bar("MAP-B"); map_b(b)
    bar("STAR SHIFTER†"); b.compose(T.shifter.inverse(), qubits=list(range(T.NQS)), inplace=True)
    bar("UNMAP-B"); map_b(b)
    bar("W2: β, 1−β, signs"); b.append(UCRYGate(list(w2_angles())), [T.W2, S_, H_, D0, D1, KIND, CQ])
    bar("UNMAP-A r→d"); map_a(b, inverse=True)
    bar("W1: 1/D(p)"); b.append(UCRYGate(list(w1a)), [T.W1] + IDX + FQ)
    bar("SUM"); b.h(LQ + [CQ])
    return b


def jacobi_step_quantum(ax, ay, az, Js, faces):
    vec, norm, w1a, invD_max = encode(ax, ay, az, Js, faces)
    psi0 = np.zeros(2**T.NQ, complex)
    for L in range(32):
        psi0[L*2**LQ[0]: L*2**LQ[0] + n**3] = vec[L]
    body = transpile(build_body(w1a), T.sim, optimization_level=0)
    qc = QuantumCircuit(T.NQ); qc.set_statevector(psi0); qc.compose(body, inplace=True); qc.save_statevector()
    sv = T.sim.run(qc).result().get_statevector().data
    out, p = {}, 0.0
    for fi, f in enumerate(FRAMES):
        ro = sv[fi*2**FQ[0]: fi*2**FQ[0] + n**3]
        p += np.sum(np.abs(ro)**2)
        full = np.zeros((n-1, n, n))
        full[T.K, T.I, T.J_] = (T.from_flat(ro).real*8*np.sqrt(6)*norm*invD_max)[T.K, T.I, T.J_]
        out[f] = full
    return out, p, w1a


def cx_of(circ):
    return transpile(circ, basis_gates=["cx", "u"], optimization_level=1).count_ops().get("cx", 0)


def named_circuits(w1a):
    regs = list(T.shifter.qregs) + [QuantumRegister(5, "L"), QuantumRegister(2, "frame"),
                                    QuantumRegister(1, "copy"), QuantumRegister(1, "w1"), QuantumRegister(1, "w2")]
    det = QuantumCircuit(*regs, name="jacobi_ctrl_pi_opt_gates")
    det.append(Gate("ENCODE products, J, a·D", len(IDX + LQ), []), IDX + LQ)
    det.compose(build_body(w1a, regs=regs, barriers=True), inplace=True)
    mp = QuantumCircuit(QuantumRegister(5, "L"), QuantumRegister(2, "frame"), QuantumRegister(1, "copy"),
                        QuantumRegister(4, "term"), name="map_opt")
    # small stand-alone picture of MAP-A + MAP-B on the relevant qubits
    sub = QuantumCircuit(T.NQ); map_a(sub); map_b(sub)
    keep = LQ + FQ + [CQ] + TQ
    for ins in sub.data:
        mp.append(ins.operation, [keep.index(sub.find_bit(x).index) for x in ins.qubits])
    return {"jacobi_ctrl_pi_opt_gates": (det, {"fold": 60, "scale": 0.6}),
            "map_opt": (mp, {"fold": -1, "scale": 0.9})}


if __name__ == "__main__":
    rng = np.random.default_rng(3)                                       # same data as test_jacobi_ctrl_pi.py
    nu_cell = rng.uniform(0.05, 1.0, (n-1, n-1, n-1))
    Fx, Fy, Fz = (T.face_family(nu_cell, a_) for a_ in (0, 1, 2))
    faces = {'z': (T.to_z(Fy), T.to_z(Fx)), 'y': (T.to_y(Fx), T.to_y(Fz)), 'x': (T.to_x(Fz), T.to_x(Fy))}
    ax = rng.standard_normal((n, n-1, n)); ay = rng.standard_normal((n, n, n-1)); az = rng.standard_normal((n-1, n, n))
    Js = {f: rng.standard_normal((n-1, n, n)) for f in FRAMES}

    t0 = time.time()
    out_q, p, w1a = jacobi_step_quantum(ax, ay, az, Js, faces)
    print(f"optimised circuit, {T.NQ} qubits: one run for all frames {time.time()-t0:.1f} s, P(success) = {p:.2e}")
    out_c = T.jacobi_step_classical(ax, ay, az, Js, faces)
    for f in FRAMES:
        print(f"  {f}-frame: max|a_new quantum - classical| = {np.abs(out_q[f] - out_c[f]).max():.1e}")

    # gate counts: old vs new MAP + UNMAP and W2
    _, _, table, _, w2_old = T.problem(ax, ay, az, Js, faces)[:5]
    old = QuantumCircuit(T.NQ); T.lookup(old, table); T.lookup(old, table)
    new = QuantumCircuit(T.NQ); map_a(new); map_b(new); map_b(new); map_a(new, inverse=True)
    mcx = lambda c: [len(i.qubits) - 1 for i in c.data if i.operation.name.startswith("mcx")]
    print(f"\nMAP + UNMAP   old: {len(mcx(old))} MCX (max {max(mcx(old))} controls), {cx_of(old)} CX")
    print(f"              new: {len(mcx(new))} MCX (max {max(mcx(new))} controls) (incl. the phase-free mod-3 steps), "
          f"{cx_of(new)} CX")
    print(f"W2 lookup     old: {len(w2_old)} angles,  new: {len(w2_angles())} angles")

    res = "results/jacobi_ctrl_pi_opt"; os.makedirs(res, exist_ok=True)
    for tag, out in (("quantum", out_q), ("classical", out_c)):
        for name, arr in zip(("ax", "ay", "az"), T.to_storage(out)):
            T.save_field_txt(f"{res}/{name}_{tag}.txt", arr,
                             f"{name} after one Jacobi step ({tag}, optimised map); interior updated, boundary 0")
    print(f"saved {res}/ax|ay|az_quantum.txt and _classical.txt")
    from circuit_saving import save_circuits
    save_circuits(named_circuits(w1a), "circuits/jacobi_ctrl_pi_opt")
