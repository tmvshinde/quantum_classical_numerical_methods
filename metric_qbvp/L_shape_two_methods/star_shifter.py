"""
Star-stencil shifter: all 12 index offsets of the curl-curl star in ONE circuit, in superposition.

Offsets needed (frame-local (dk, di, dj), see star_term_shifts.md section 5b):
    E: (0,+1,0) (0,0,0) (+1,0,0)        W: (0,-1,0) (0,-1,0) (+1,-1,0)
    N: (0,0,+1) (0,0,0) (+1,0,0)        S: (0,0,-1) (0,0,-1) (+1,0,-1)

Design ("offset-factorised select"):
  1. Term register  |alpha, sigma, r1, r0>  chosen so the shift controls are (almost) single qubits:
       alpha = axis (0: E/W -> i,  1: N/S -> j)     sigma = side (0: E/N,  1: W/S)
       member r = (r1, r0):  'a' = 01,  'b/c at k' = 00,  'b/c at k+1' = 10
     Then  dk = r1,   axis shift active  s = sigma OR r0,   direction (minus) = sigma.
  2. Per index group one ripple-carry incrementer with the control flag as CARRY-IN
     (a controlled increment costs the same as an uncontrolled one), built from
     temporary logical-ANDs (4 T to compute, 0 T to uncompute by measurement).
  3. Direction by the complement trick:  x - 1 = NOT(NOT(x) + 1), with CNOTs from sigma (Clifford).
  4. Carry ancillas are reused across the three groups.

Baselines:
  B1  naive: each of the 12 shift operations controlled on its full 4-qubit term code
  B2  per-axis flags (same as ours) but the textbook MCX-cascade increment
  B3  QFT-based: per axis QFT, flag/direction-controlled phase layer, inverse QFT

The script verifies the proposed circuit (and B2) on every basis input, then prints resource counts.
"""
import itertools
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate


# ------------------------------------------------------------------------------------------------
# term codes (alpha, sigma, r1, r0) and the offsets they must produce
FACES = {"E": (0, 0), "W": (0, 1), "N": (1, 0), "S": (1, 1)}
MEMBERS = {"a": (0, 1), "bc_k": (0, 0), "bc_k1": (1, 0)}

def expected_offset(alpha, sigma, r1, r0):
    dk = r1
    d_axis = -1 if sigma else (+1 if r0 else 0)
    return (dk, d_axis, 0) if alpha == 0 else (dk, 0, d_axis)

TERMS = [(f, m, *FACES[f], *MEMBERS[m]) for f in "EWNS" for m in MEMBERS]


# ------------------------------------------------------------------------------------------------
class Counter:
    """Fault-tolerant cost model: temporary AND = 4 T (compute), 0 T (measurement-based uncompute)."""
    def __init__(self): self.and_c = 0; self.and_u = 0
    @property
    def T(self): return 4*self.and_c


def AND(qc, cnt, a, b, t):            # compute temporary AND  t = a & b   (t starts in |0>)
    qc.ccx(a, b, t); cnt.and_c += 1

def AND_dg(qc, cnt, a, b, t):         # uncompute (unitary stand-in for measurement + CZ fix-up)
    qc.ccx(a, b, t); cnt.and_u += 1


def inc_carry_in(qc, cnt, g, cin, carry):
    """g <- g + cin (mod 2^q). Ripple carry with temporary ANDs; carry has q-1 clean qubits."""
    q = len(g)
    if q == 1:
        qc.cx(cin, g[0]); return
    AND(qc, cnt, cin, g[0], carry[0])                       # c0 = cin & g0
    for m in range(1, q - 1):
        AND(qc, cnt, carry[m-1], g[m], carry[m])            # c_m = c_{m-1} & g_m
    for m in range(q - 1, 0, -1):                           # top-down: flip, then release carry
        qc.cx(carry[m-1], g[m])                             # g_m ^= c_{m-1}
        a = cin if m == 1 else carry[m-2]
        AND_dg(qc, cnt, a, g[m-1], carry[m-1])              # g_{m-1} not flipped yet -> valid
    qc.cx(cin, g[0])                                        # g_0 ^= cin


def build_proposed(q):
    gk, gi, gj = QuantumRegister(q, "gk"), QuantumRegister(q, "gi"), QuantumRegister(q, "gj")
    tm = QuantumRegister(4, "term")                         # alpha, sigma, r1, r0
    fl = QuantumRegister(3, "flag")                         # s, fi, fj
    ca = QuantumRegister(max(q - 1, 1), "carry")
    qc = QuantumCircuit(gk, gi, gj, tm, fl, ca)
    cnt = Counter()
    alpha, sigma, r1, r0 = tm
    s, fi, fj = fl

    # flags: s = sigma OR r0 ; fi = s AND NOT alpha ; fj = s XOR fi (= s AND alpha)
    qc.x([sigma, r0]); AND(qc, cnt, sigma, r0, s); qc.x([sigma, r0, s])
    qc.x(alpha); AND(qc, cnt, s, alpha, fi); qc.x(alpha)
    qc.cx(s, fj); qc.cx(fi, fj)

    inc_carry_in(qc, cnt, list(gk), r1, list(ca))           # k: +1 when r1
    for g, f in [(gi, fi), (gj, fj)]:                       # i, j: +-1 with complement trick
        for b in g: qc.cx(sigma, b)
        inc_carry_in(qc, cnt, list(g), f, list(ca))
        for b in g: qc.cx(sigma, b)

    # uncompute flags
    qc.cx(fi, fj); qc.cx(s, fj)
    qc.x(alpha); AND_dg(qc, cnt, s, alpha, fi); qc.x(alpha)
    qc.x([sigma, r0, s]); AND_dg(qc, cnt, sigma, r0, s); qc.x([sigma, r0])
    return qc, cnt, (gk, gi, gj, tm, fl, ca)


def build_B2(q):
    """Same flags, but MCX-cascade increment (controls = flag + lower bits)."""
    gk, gi, gj = QuantumRegister(q, "gk"), QuantumRegister(q, "gi"), QuantumRegister(q, "gj")
    tm = QuantumRegister(4, "term"); fl = QuantumRegister(3, "flag")
    anc = QuantumRegister(max(q - 2, 1), "anc")
    qc = QuantumCircuit(gk, gi, gj, tm, fl, anc)
    alpha, sigma, r1, r0 = tm; s, fi, fj = fl
    qc.x([sigma, r0]); qc.ccx(sigma, r0, s); qc.x([sigma, r0, s])
    qc.x(alpha); qc.ccx(s, alpha, fi); qc.x(alpha); qc.cx(s, fj); qc.cx(fi, fj)
    def cascade(g, c):
        for m in range(q - 1, -1, -1):
            ctrls = [c] + list(g[:m])
            if len(ctrls) == 1: qc.cx(c, g[m])
            elif len(ctrls) == 2: qc.ccx(*ctrls, g[m])
            else: qc.mcx(ctrls, g[m])        # synthesis left to the transpiler (may use idle qubits)
    cascade(gk, r1)
    for g, f in [(gi, fi), (gj, fj)]:
        for b in g: qc.cx(sigma, b)
        cascade(g, f)
        for b in g: qc.cx(sigma, b)
    qc.cx(fi, fj); qc.cx(s, fj); qc.x(alpha); qc.ccx(s, alpha, fi); qc.x(alpha)
    qc.x([sigma, r0, s]); qc.ccx(sigma, r0, s); qc.x([sigma, r0])
    return qc


def build_B3_qft(q):
    """QFT-based: per axis F^-1 P(+-1, flag) F. CX count only (rotations are non-Clifford)."""
    g = QuantumRegister(q, "g"); c = QuantumRegister(1, "flag"); d = QuantumRegister(1, "dir")
    fd = QuantumRegister(1, "flag_and_dir")
    qc = QuantumCircuit(g, c, d, fd)
    qc.append(QFTGate(q), g)
    for l in range(q):               # +1 shift: phase 2*pi*2^l/2^q on qubit l (Qiskit QFT convention)
        th = -2*np.pi*2**l/2**q
        qc.cp(th, c[0], g[l]); qc.cp(-2*th, fd[0], g[l])     # sign flip when flag & dir
    qc.append(QFTGate(q).inverse(), g)
    return qc


# ------------------------------------------------------------------------------------------------
def simulate_bits(qc, bits):
    """Classical simulation of a reversible circuit (x, cx, ccx, mcx) on a dict qubit->bit."""
    for ins in qc.data:
        name = ins.operation.name
        qs = ins.qubits
        if name == "x":
            bits[qs[0]] ^= 1
        elif name in ("cx", "ccx", "mcx"):
            if all(bits[x] for x in qs[:-1]): bits[qs[-1]] ^= 1
        elif name == "barrier":
            pass
        else:
            raise ValueError(name)
    return bits


def verify(qc, regs, q, mcx_ok=False):
    gk, gi, gj, tm, fl, ca = regs
    ok = True; n = 2**q
    for (face, mem, al, sg, r1, r0) in TERMS:
        dk, di, dj = expected_offset(al, sg, r1, r0)
        for pk, pi_, pj in itertools.product(range(n), repeat=3):
            bits = {x: 0 for x in qc.qubits}
            for reg, val in [(gk, pk), (gi, pi_), (gj, pj)]:
                for b in range(q): bits[reg[b]] = (val >> b) & 1
            for x, v in zip(tm, (al, sg, r1, r0)): bits[x] = v
            out = simulate_bits(qc, bits)
            get = lambda reg: sum(out[reg[b]] << b for b in range(q))
            res = (get(gk), get(gi), get(gj))
            exp = ((pk + dk) % n, (pi_ + di) % n, (pj + dj) % n)
            clean = all(out[x] == 0 for x in list(fl) + list(ca))
            term_ok = tuple(out[x] for x in tm) == (al, sg, r1, r0)
            if res != exp or not clean or not term_ok:
                ok = False
    return ok


def cx_count(qc):
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    return t.count_ops().get("cx", 0)


if __name__ == "__main__":
    print("verification on every basis input (12 terms x all index values):")
    for q in (1, 2, 3, 4):
        qc, cnt, regs = build_proposed(q)
        print(f"  q = {q}: proposed {'OK' if verify(qc, regs, q) else 'FAIL'}")

    # B2 verification (decompose mcx for simulation through qiskit's own gates is not needed:
    # mcx semantics are simulated directly, but v-chain ancillas must stay clean) -> check q=3,4
    for q in (3, 4):
        qc2 = build_B2(q)
        regs2 = tuple(qc2.qregs)
        print(f"  q = {q}: B2 (MCX cascade) {'OK' if verify(qc2, regs2, q) else 'FAIL'}")

    print("\nresources for the full 12-term star shift (three q-qubit index groups):")
    print(f"{'q':>3} | {'proposed T':>10} {'B2 T':>8} {'B1 T':>8} | {'proposed CX':>11} {'B2 CX':>7} {'B3 QFT CX':>10} "
          f"| {'QFT rot.':>8} | {'anc prop.':>9}")
    for q in range(2, 11):
        qc, cnt, regs = build_proposed(q)
        T_prop = cnt.T
        T_B2 = 8 + 3*2*q*(q - 1)                          # 4m T for bit m (m+1 controls), 3 axes
        T_B1 = 12*4*(q*(q - 1)//2 + 3*q)                  # 12 shifts, 4 extra controls each
        cx_prop = cx_count(qc)
        cx_B2 = cx_count(build_B2(q))
        cx_B3 = 3*cx_count(build_B3_qft(q))
        rot_B3 = 3*(q*(q - 1) + 2*q)                       # controlled phases (each 1 non-Clifford rot. pair)
        anc = (q - 1) + 3
        print(f"{q:>3} | {T_prop:>10} {T_B2:>8} {T_B1:>8} | {cx_prop:>11} {cx_B2:>7} {cx_B3:>10} | {rot_B3:>8} | {anc:>9}")
    print("\nT counts: temporary-AND model (4 T compute, 0 T measurement-based uncompute).")
    print("CX counts: Qiskit transpile to {cx, u}, unitary uncompute (an upper bound for 'proposed').")
