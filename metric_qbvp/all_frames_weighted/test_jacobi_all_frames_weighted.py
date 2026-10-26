"""
Jacobi step for all three frames in ONE circuit:

    a_new = beta * ( S + J ) / D  +  (1 - beta) * a ,      S = sum of the 12 star terms coeff_nf * field
    (paper_section/product_blocks.tex, Eq. (jacobi): A_z <- beta/Delta (E + W + N + S + J_z) + (1 - beta) A_z)

Encoded (per frame f, in its own frame layout; frame register |f> labels the frame):
    type 0 : the 12 star blocks  coeff_nf * field / D(target edge)     (term codes 0..11)
    type 1 : J / D                                                    (term code 0000: not shifted)
    type 2 : a                                                        (term code 0000: not shifted)
Circuit:
    SIGN (star terms)  ->  STAR SHIFTER^dag (all 12 shifts, all frames at once)
    -> constants on a weight ancilla w:  RY(2 acos(beta)) if type in {0,1},  RY(2 acos(1-beta)) if type = 2
       (+J: the J subspace gets +beta like the star terms, no sign gate)
    -> H on term (4) + type (2)  -> sum onto |term = 0, type = 0>
    read <frame f, w=0, type=0, term=0, rest=0, p | Psi> * 8 * N = a_new of frame f
"""
import time, math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator
from star_shifter import build_proposed, TERMS, expected_offset
import test_combined_frames_aer as C

q, n, beta = C.q, C.n, C.beta
shifter, _, _ = build_proposed(q)
NQS = shifter.num_qubits                                   # 18
IDX = list(range(3*q)); TQ = [3*q + t for t in range(4)]
TY = [NQS, NQS + 1]; W = NQS + 2; FR = [NQS + 3, NQS + 4]
NQ = NQS + 5                                               # 23
sim = AerSimulator(method="statevector")


def sign_star(c, t):                                       # minus on E_bc_k1, W_bc_k, N_bc_k1, S_bc_k
    c.x(t[1]); c.cz(t[1], t[2]); c.x(t[1])
    c.x([t[2], t[3]]); c.ccz(t[1], t[2], t[3]); c.x([t[2], t[3]])


def constants(c, ty, w):                                   # beta on type 0, 1 (ty1 = 0); 1 - beta on type 2
    c.append(RYGate(2*math.acos(beta)).control(1, ctrl_state=0), [ty[1], w])
    c.append(RYGate(2*math.acos(1 - beta)).control(1, ctrl_state=1), [ty[1], w])


def build_body(regs=None):
    b = QuantumCircuit(*regs) if regs else QuantumCircuit(NQ)
    sign_star(b, TQ)
    b.compose(shifter.inverse(), qubits=list(range(NQS)), inplace=True)
    constants(b, TY, W)
    b.h(TQ + TY)
    return b


body_t = transpile(build_body(), sim, optimization_level=0)


def encode(ax, ay, az, Js):
    """amp[frame, type, term, index]: star blocks / D(target), J / D, a (each in its frame layout)"""
    fr = C.frames_of(ax, ay, az)
    amp = np.zeros((3, 4, 16, n**3))
    for fi, f in enumerate(C.FRAMES):
        a, b, c = fr[f]; nfy, nfx = C.faces[f]
        Q = C.blocks(a, b, c, nfy, nfx)
        D_full = np.zeros((n, n, n)); D_full[C.K, C.I, C.J_] = C.diag_D(nfy, nfx)
        inside = D_full > 0
        invD = np.where(inside, 1/np.where(inside, D_full, 1), 0.0)
        for (face, mem, al, sg, r1, r0) in TERMS:
            blk = C.pad(Q[C.TERM_BLOCK[(face, mem)]])*np.roll(invD, expected_offset(al, sg, r1, r0), axis=(0, 1, 2))
            amp[fi, 0, al + 2*sg + 4*r1 + 8*r0] = C.to_flat(blk)
        amp[fi, 1, 0] = C.to_flat(C.pad(Js[f])*invD)                 # J / D
        amp[fi, 2, 0] = C.to_flat(C.pad(a))                           # a
    return amp, np.linalg.norm(amp)


def jacobi_step_all_frames(ax, ay, az, Js):
    """one circuit run -> a_new (interior, frame layout) for z, y, x"""
    amp, norm = encode(ax, ay, az, Js)
    psi0 = np.zeros(2**NQ, complex)
    for fi in range(3):
        for t in range(3):
            base = fi*2**FR[0] + t*2**TY[0]
            psi0[base: base + 16*n**3] = amp[fi, t].ravel()/norm
    qc = QuantumCircuit(NQ); qc.set_statevector(psi0)
    qc.compose(body_t, inplace=True); qc.save_statevector()
    sv = sim.run(qc).result().get_statevector().data
    out, p = {}, 0.0
    for fi, f in enumerate(C.FRAMES):
        ro = sv[fi*2**FR[0]: fi*2**FR[0] + n**3]             # w=0, type=0, term=0, flags/carries=0
        out[f] = (C.from_flat(ro).real*8*norm)[C.K, C.I, C.J_]
        p += np.sum(np.abs(ro)**2)
    return out, p


def classical(a, b, c, J, nfy, nfx):
    return beta*(C.star_sum_classical(a, b, c, nfy, nfx) + J[C.K, C.I, C.J_])/C.diag_D(nfy, nfx) \
           + (1 - beta)*a[C.K, C.I, C.J_]


def named_circuits():
    """block-level and gate-level circuits with named registers, for saving"""
    from qiskit.circuit import Gate
    regs = list(shifter.qregs) + [QuantumRegister(2, "type"), QuantumRegister(1, "w"), QuantumRegister(2, "frame")]
    enc = IDX + TQ + TY + FR
    blk = QuantumCircuit(*regs, name="jacobi_all_frames_blocks")
    blk.append(Gate("ENCODE coeff·a/D, J/D, a", len(enc), []), enc)
    s = QuantumCircuit(4); sign_star(s, [0, 1, 2, 3]); blk.append(s.to_gate(label="SIGN star"), TQ)
    blk.append(shifter.inverse().to_gate(label="STAR SHIFTER†"), list(range(NQS)))
    cw = QuantumCircuit(3); constants(cw, [0, 1], 2); blk.append(cw.to_gate(label="β, 1−β"), TY + [W])
    hs = QuantumCircuit(6); hs.h(range(6)); blk.append(hs.to_gate(label="SUM H⊗6"), TQ + TY)
    det = QuantumCircuit(*regs, name="jacobi_all_frames_gates")
    det.append(Gate("ENCODE coeff·a/D, J/D, a", len(enc), []), enc)
    det.barrier(label="SIGN"); sign_star(det, TQ)
    det.barrier(label="STAR SHIFTER†"); det.compose(shifter.inverse(), qubits=list(range(NQS)), inplace=True)
    det.barrier(label="β, 1−β"); constants(det, TY, W)
    det.barrier(label="SUM"); det.h(TQ + TY)
    return {"jacobi_all_frames_blocks": (blk, {"fold": -1, "scale": 0.8}),
            "jacobi_all_frames_gates": (det, {"fold": 50, "scale": 0.65})}


def apply_bc_az(az, A_left=1e-2, A_right=5e-2):
    az[:, 0, :] = A_left; az[:, -1, :] = A_right
    az[:, 1:-1, 0] = az[:, 1:-1, 1]; az[:, 1:-1, -1] = az[:, 1:-1, -2]
    return az


def solve(step, timesteps=2000, conv_tol=1e-7):
    """L-shape benchmark (J = 0), boundary conditions as in the notebooks"""
    ax = np.zeros((n, n-1, n)); ay = np.zeros((n, n, n-1)); az = apply_bc_az(np.zeros((n-1, n, n)))
    Js = {f: np.zeros((n-1, n, n)) for f in C.FRAMES}
    for t in range(timesteps):
        new = step(ax, ay, az, Js)
        full = {}
        for f in C.FRAMES:
            arr = np.zeros((n-1, n, n)); arr[C.K, C.I, C.J_] = new[f]; full[f] = arr
        axn, ayn, azn = C.xyz_to_zxy(full['x']), C.yzx_to_zxy(full['y']), apply_bc_az(full['z'])
        err = np.linalg.norm(azn - az)**2
        ax, ay, az = axn, ayn, azn
        if err < conv_tol:
            break
    return az, t


def step_quantum(ax, ay, az, Js):
    return jacobi_step_all_frames(ax, ay, az, Js)[0]


def step_classical(ax, ay, az, Js):
    fr = C.frames_of(ax, ay, az)
    return {f: classical(*fr[f], Js[f], *C.faces[f]) for f in C.FRAMES}


if __name__ == "__main__":
    print(f"{NQ} qubits: {NQS} star shifter + type(2) + w(1) + frame(2);  beta = {beta}\n")
    rng = np.random.default_rng(0)
    tax = rng.standard_normal((n, n-1, n)); tay = rng.standard_normal((n, n, n-1)); taz = rng.standard_normal((n-1, n, n))
    Js = {f: rng.standard_normal((n-1, n, n)) for f in C.FRAMES}
    fr = C.frames_of(tax, tay, taz)
    t0 = time.time(); out, p = jacobi_step_all_frames(tax, tay, taz, Js); dt = time.time() - t0
    print(f"one Jacobi step, random fields and J, ONE circuit run for all frames ({dt:.1f} s, P(success) = {p:.4f}):")
    for f in C.FRAMES:
        err = np.abs(out[f] - classical(*fr[f], Js[f], *C.faces[f])).max()
        print(f"  {f}-frame: max|a_new quantum - classical| = {err:.1e}")

    t0 = time.time(); azq, tq_ = solve(step_quantum); tq_time = time.time() - t0
    azc, tc_ = solve(step_classical)
    print(f"\nfull L-shape solve: quantum converged at step {tq_} ({tq_time:.0f} s, {tq_ + 1} circuit runs), "
          f"classical at step {tc_};  max|az_quantum - az_classical| = {np.abs(azq - azc).max():.1e}")

    from circuit_saving import save_circuits
    save_circuits(named_circuits(), "circuits/jacobi_all_frames_weighted")
