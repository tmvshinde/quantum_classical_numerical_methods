from os import stat
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.circuit.library import (
    standard_gates as gates,
    MCXGate,
    RYGate,
    MCMTGate,
    ZGate,
    SwapGate,
)

import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import math

np.set_printoptions(threshold=sys.maxsize, linewidth=200)


def streaming_0_3(qc, ctrl_string, fix_ctrl_q, N, num_q_d):

    ctrlstate = ctrl_string

    len_ctrlstate = len(ctrlstate)

    num_q_dir = (N - len_ctrlstate) // 3
    # print("num_q_dir: ", num_q_dir)

    max_num_ctrls = len_ctrlstate + num_q_dir - 1
    min_num_ctrls = len_ctrlstate + 2

    for num_ctrls in range(max_num_ctrls, min_num_ctrls - 1, -1):
        # print("num_ctrls: ", num_ctrls)

        prefix = "1" * (num_ctrls - len_ctrlstate)
        ctrlstatestr = prefix + ctrlstate
        # print("ctrlstatestr: ", ctrlstatestr)

        ctrl_qubits = fix_ctrl_q + list(range(num_q_d, num_q_d + len(prefix)))
        # + list(range(N-8, N-8 - (num_ctrls - 7), -1))
        # print("ctrl_qubits: ", ctrl_qubits)

        target_qubit = num_q_d + num_ctrls - min_num_ctrls + 2
        # print("target_qubit: ", target_qubit)

        all_qubits = ctrl_qubits + [target_qubit]

        qc.append(MCXGate(num_ctrls, ctrl_state=ctrlstatestr), all_qubits)

    qc.append(MCXGate(len_ctrlstate, ctrl_state=ctrlstate), fix_ctrl_q + [num_q_d])
    qc.append(MCXGate(len_ctrlstate, ctrl_state=ctrlstate), fix_ctrl_q + [num_q_d + 1])
    qc.append(
        MCXGate(len_ctrlstate + 1, ctrl_state="1" + ctrlstate),
        fix_ctrl_q + [num_q_d, num_q_d + 1],
    )

    return qc


def streaming_1(qc, ctrl_string, fix_ctrl_q, N, num_q_d):

    ctrlstate = ctrl_string
    len_ctrlstate = len(ctrlstate)

    num_q_dir = (N - len_ctrlstate) // 3

    max_num_ctrls = len_ctrlstate + num_q_dir - 1
    min_num_ctrls = len_ctrlstate + 2

    qc.append(MCXGate(len_ctrlstate, ctrl_state=ctrlstate), fix_ctrl_q + [num_q_d])
    qc.append(
        MCXGate(len_ctrlstate + 1, ctrl_state="1" + ctrlstate),
        fix_ctrl_q + [num_q_d, num_q_d + 1],
    )
    # qc.append(MCXGate(7, ctrl_state=ctrlstate), [ N- 1, N-2, N-3, N-4, N-5, N-6, N-7, 2])

    for num_ctrls in range(min_num_ctrls, max_num_ctrls + 1):
        # print("num_ctrls: ", num_ctrls)

        prefix = "1" * (num_ctrls - len_ctrlstate)
        ctrlstatestr = prefix + ctrlstate
        # print("ctrlstatestr: ", ctrlstatestr)

        ctrl_qubits = fix_ctrl_q + list(range(num_q_d, num_q_d + len(prefix)))
        # + list(range(N-8, N-8 - (num_ctrls - 7), -1))
        # print("ctrl_qubits: ", ctrl_qubits)

        target_qubit = num_q_d + num_ctrls - min_num_ctrls + 2
        # print("target_qubit: ", target_qubit)

        all_qubits = ctrl_qubits + [target_qubit]

        qc.append(
            MCXGate(len_ctrlstate, ctrl_state=ctrlstate), fix_ctrl_q + [target_qubit]
        )
        # print("done 1")
        qc.append(MCXGate(num_ctrls, ctrl_state=ctrlstatestr), all_qubits)
        # print("done 2")
        qc.append(
            MCXGate(len_ctrlstate, ctrl_state=ctrlstate), fix_ctrl_q + [target_qubit]
        )

    return qc


def streaming_azz(total_qubits, numqx, numqz, ctrls_az, ctrls_copies):

    # these streamings are for calculating az values with beta 1
    qc = QuantumCircuit(total_qubits + 1)
    numqx = numqx
    numqz = numqz
    ctrls_states = []
    len_ctrls_copies = len(ctrls_copies)
    for i in ctrls_copies:
        # print(i[::-1])
        # ctrls_states.append(ctrls_az + i[::-1] + "0")
        ctrls_states.append(ctrls_az + i[::-1] + "00")

    print(ctrls_states)

    len_ctrlstate = len(ctrls_states[0])
    # fix_ctrl_q = [total_qubits - i for i in range(0, len_ctrlstate - 2)] + [
    #     total_qubits - len_ctrlstate - 1,
    #     total_qubits - len_ctrlstate - 2,
    # ]

    fix_ctrl_q = [total_qubits ,total_qubits-1,total_qubits-2 , total_qubits - 3, total_qubits - 4] + [
        total_qubits - len_ctrlstate - 1,
        total_qubits - len_ctrlstate - 2,
    ]

    qc = streaming_1(
        qc, ctrls_states[0], fix_ctrl_q, total_qubits -1, numqx
    )  # az[k,i+1,j]
    qc = streaming_0_3(
        qc, ctrls_states[1], fix_ctrl_q, total_qubits -1, numqx
    )  # az[k,i-1,j]
    qc = streaming_1(
        qc, ctrls_states[2], fix_ctrl_q, total_qubits -1, numqz
    )  # az[k,i,j+1]
    qc = streaming_0_3(
        qc, ctrls_states[3], fix_ctrl_q, total_qubits -1, numqz
    )  # az[k,i,j-1]

    return qc


def streaming_axz(total_qubits, numqx, numqy, ctrls_ax, ctrls_copies):

    # nothing for ax[k,i,j] as it is already in the correct position numqx
    # these streamings are for calculating az values
    qc = QuantumCircuit(total_qubits + 1)
    ctrls_states = []
    len_ctrls_copies = len(ctrls_copies)
    for i in ctrls_copies:
        # print(i[::-1])
        ctrls_states.append(ctrls_ax + i[::-1] + "00")
        # ctrls_states.append(ctrls_ax + i[::-1])

    # print(ctrls_states)
    len_ctrl_states = len(ctrls_states[0])

    # fix_ctrl_q = [total_qubits - i for i in range(0, len_ctrl_states - 2)] + [
    #     total_qubits - len_ctrl_states - 1,
    #     total_qubits - len_ctrl_states - 2,
    # ]

    fix_ctrl_q = [total_qubits ,total_qubits-1,total_qubits-2 , total_qubits - 3, total_qubits - 4] + [
        total_qubits - len_ctrl_states - 1,
        total_qubits - len_ctrl_states - 2,
    ]

    # print(len(fix_ctrl_q))
    qc = streaming_1(
        qc, ctrls_states[1], fix_ctrl_q, total_qubits -1, numqy
    )  # ax[k+1,i,j]
    for i in range(2):
        qc.append(
            ZGate().control(len_ctrl_states, ctrl_state=ctrls_states[i + 1]),
            fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
        )  # -ve ax[k+1,i,j]
        qc.append(
            MCXGate(len_ctrl_states, ctrl_state=ctrls_states[i + 1]),
            fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
        )  # swap ax[k+1,i,j] with numqx
        qc.append(
            ZGate().control(len_ctrl_states, ctrl_state=ctrls_states[i + 1]),
            fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
        )  # -ve ax[k+1,i,j]
        qc.append(
            MCXGate(len_ctrl_states, ctrl_state=ctrls_states[i + 1]),
            fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
        )  # swap ax[k+1,i,j] with numqx

    ctrls_for_padding = "1" * (numqx - 1)
    qubits_for_padding = [i for i in range(numqy - 1, numqx - 1, -1)]
    len_ctrl_padding = len(ctrls_for_padding)
    # print(ctrls_for_padding, qubits_for_padding)

    qc.append(
        MCXGate(
            len_ctrl_states + len_ctrl_padding,
            ctrl_state=ctrls_for_padding + ctrls_states[2],
        ),
        fix_ctrl_q + qubits_for_padding,
    )
    qc = streaming_0_3(
        qc, ctrls_states[2], fix_ctrl_q, total_qubits -1, numqx
    )  # ax[k,i-1,j]

    qc.append(
        MCXGate(
            len_ctrl_states + len_ctrl_padding,
            ctrl_state=ctrls_for_padding + ctrls_states[3],
        ),
        fix_ctrl_q + qubits_for_padding,
    )
    qc = streaming_1(
        qc, ctrls_states[3], fix_ctrl_q, total_qubits -1, numqy
    )  # ax[k+1,i-1,j] k+1
    qc = streaming_0_3(
        qc, ctrls_states[3], fix_ctrl_q, total_qubits -1, numqx
    )  # ax[k+1,i-1,j] i -1

    return qc


def streaming_ayz(total_qubits, numqy, numqz, ctrls_ay, ctrls_copies):

    # these streamings are for calculating az values
    qc = QuantumCircuit(total_qubits + 1)
    numqx = numqy // 2
    half = numqy // 2
    ctrls_states = []
    # print("numqy", numqy)
    len_ctrls_copies = len(ctrls_copies)
    for i in ctrls_copies:
        # print(i[::-1])
        ctrls_states.append(ctrls_ay + i[::-1] + "00")
        # ctrls_states.append(ctrls_ay + i[::-1])

    len_ctrl_states = len(ctrls_states[0])

    # fix_ctrl_q = [total_qubits - i for i in range(0, len_ctrl_states - 2)] + [
    #     total_qubits - len_ctrl_states - 1,
    #     total_qubits - len_ctrl_states - 2,
    # ]

    fix_ctrl_q = [total_qubits ,total_qubits-1, total_qubits - 2, total_qubits - 3, total_qubits - 4] + [
        total_qubits - len_ctrl_states - 1,
        total_qubits - len_ctrl_states - 2,
    ]

    qc.append(
        ZGate().control(len_ctrl_states, ctrl_state=ctrls_states[0]),
        fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
    )  # -ve ax[k+1,i,j]
    qc.append(
        MCXGate(len_ctrl_states, ctrl_state=ctrls_states[0]),
        fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
    )  # swap ax[k+1,i,j] with numqx
    qc.append(
        ZGate().control(len_ctrl_states, ctrl_state=ctrls_states[0]),
        fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
    )  # -ve ax[k+1,i,j]
    qc.append(
        MCXGate(len_ctrl_states, ctrl_state=ctrls_states[0]),
        fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
    )  # swap ax[k+1,i,j] with numqx

    ctrls_for_padding = "1" * (numqx - 1)
    qubits_for_padding = [i for i in range(numqx - 1, -1, -1)]
    len_ctrl_padding = len(ctrls_for_padding)
    # print(ctrls_for_padding, qubits_for_padding)
    qc.append(
        MCXGate(
            len_ctrl_states + len_ctrl_padding,
            ctrl_state=ctrls_for_padding + ctrls_states[0],
        ),
        fix_ctrl_q + qubits_for_padding,
    )
    qc = streaming_0_3(
        qc, ctrls_states[0], fix_ctrl_q, total_qubits -2, numqz
    )  # ay[k,i,j-1

    swapgate1 = SwapGate().control(len_ctrl_states, ctrl_state=ctrls_states[0])

    qc.append(
        MCXGate(
            len_ctrl_states + len_ctrl_padding,
            ctrl_state=ctrls_for_padding + ctrls_states[1],
        ),
        fix_ctrl_q + qubits_for_padding,
    )
    qc = streaming_1(
        qc, ctrls_states[1], fix_ctrl_q, total_qubits -2, numqy
    )  # ay[k+1,i,j-1]
    qc = streaming_0_3(
        qc, ctrls_states[1], fix_ctrl_q, total_qubits , numqz
    )  # ay[k+1,i,j-1]

    qc.append(
        ZGate().control(len_ctrl_states, ctrl_state=ctrls_states[3]),
        fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
    )  # -ve ax[k+1,i,j]
    qc.append(
        MCXGate(len_ctrl_states, ctrl_state=ctrls_states[3]),
        fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
    )  # swap ax[k+1,i,j] with numqx
    qc.append(
        ZGate().control(len_ctrl_states, ctrl_state=ctrls_states[3]),
        fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
    )  # -ve ax[k+1,i,j]
    qc.append(
        MCXGate(len_ctrl_states, ctrl_state=ctrls_states[3]),
        fix_ctrl_q + [total_qubits - len_ctrl_states - 3],
    )  # swap ax[k+1,i,j] with numqx

    qc = streaming_1(
        qc, ctrls_states[3], fix_ctrl_q, total_qubits -2, numqy
    )  # ay[k+1,i,j]

    return qc


def relaxation_param_pi(state_ctl, angle_p, num_total_qubits):
    # totalqsrp= num_total_qubits-8
    qc = QuantumCircuit(num_total_qubits+1)
    num_controls = len(state_ctl[0])
    # qc.cx(num_total_qubits - 8, num_total_qubits - 1)

    qc.append(
        MCXGate(
            2,
            ctrl_state="01",
        ),
        [num_total_qubits - 9,num_total_qubits, num_total_qubits - 1],
    )

    for angle, state in zip(angle_p, state_ctl):
        CR = gates.RYGate(2 * angle).control(num_controls, ctrl_state=state)
        # qc.append(CR, [num_total_qubits - 1, num_total_qubits - 2,  num_total_qubits -6])
        qc.append(
            CR,  
            [num_total_qubits - i for i in range(0, num_controls )]
            + [num_total_qubits - 9],
        )

    qc.append(
        SwapGate().control(1, ctrl_state="0"),
        [num_total_qubits, num_total_qubits - 9, num_total_qubits - 1],
    )
    # qc.swap(num_total_qubits - 7, num_total_qubits - 1)

    return qc

def relaxation_param(state_ctl, angle_p, num_total_qubits):
    # totalqsrp= num_total_qubits-8
    qc = QuantumCircuit(num_total_qubits)
    num_controls = len(state_ctl[0])
    qc.cx(num_total_qubits - 9, num_total_qubits - 1)
    for angle, state in zip(angle_p, state_ctl):
        CR = gates.RYGate(2 * angle).control(num_controls, ctrl_state=state)
        # qc.append(CR, [num_total_qubits - 1, num_total_qubits - 2,  num_total_qubits -6])
        # qc.append(
        #     CR,
        #     [num_total_qubits - i for i in range(1, num_controls + 1)]
        #     + [num_total_qubits - 9],
        # )

    # qc.swap(num_total_qubits - 9, num_total_qubits - 1)

    return qc


def summations_pi(total_qubits, num_qubits):

    num_total_qubits = total_qubits + 1
    qc = QuantumCircuit(num_total_qubits+1)

    # qc.swap(total_qubits - 7, total_qubits)
    qc.append(
        SwapGate().control(1, ctrl_state="0"),
        [num_total_qubits , num_total_qubits - 9, num_total_qubits - 1],
    )

    # qc.ch(num_total_qubits, total_qubits - 2 , ctrl_state="0")
    qc.h( total_qubits - 2)

    qc.append(
        SwapGate().control(3, ctrl_state="000"),
        [num_total_qubits , total_qubits, total_qubits - 1, total_qubits - 8, total_qubits - 7],
    )
    
    qc.barrier()

    qc.append(
        SwapGate().control(2, ctrl_state="00"),
        [num_total_qubits, total_qubits, total_qubits - 2, total_qubits - 3],
    )
    # # qc.append(MCXGate(5, ctrl_state="00100"), [total_qubits - i for i in range(0, 5)] + [total_qubits - 5]) # -ve ax[k+1,i,j]
    qc.append(
        SwapGate().control(2, ctrl_state="10"),
        [num_total_qubits, total_qubits, total_qubits - 2, total_qubits - 3],
    )
    
    # # # qc.append(MCXGate(6, ctrl_state="000001"), [total_qubits - i for i in range(0, 6)] + [total_qubits - 6]) # -ve ax[k+1,i,j]
    # qc.ch(num_total_qubits, total_qubits - 2 , ctrl_state="0")
    qc.h( total_qubits - 2)
    
    # qc.barrier()

    qc.append(
        SwapGate().control(2, ctrl_state="10"),
        [num_total_qubits, total_qubits, total_qubits - 1, total_qubits - 7],
    )
    angles = [math.acos(1 / 4), math.asin(1 / 4)]
    # # qc.barrier()
    state_ctl2 = ["110"]
    for angle, state in zip(angles, state_ctl2):
        CR = gates.RYGate(2 * angle).control(3, ctrl_state=state)

        qc.append(CR, [num_total_qubits, num_total_qubits - 1, num_total_qubits - 2] + [num_qubits])
    qc.barrier()

    qc.append(
        MCXGate(3, ctrl_state="100"),
        [num_total_qubits, total_qubits, total_qubits - 1] + [total_qubits - 7],
    )  # -ve ax[k+1,i,j]
    state_ctl3 = ["100"]
    # qc.append(MCXGate(6, ctrl_state="000000"), [total_qubits - i for i in range(0, 6)] + [total_qubits - 6]) # -ve ax[k+1,i,j]
    for angle, state in zip(angles, state_ctl3):
        CR = gates.RYGate(2 * angle).control(3, ctrl_state=state)

        qc.append(CR, [num_total_qubits, num_total_qubits - 1, num_total_qubits - 2] + [num_qubits + 1])

    qc.append(
        MCXGate(4, ctrl_state="1000"),
        [num_total_qubits, total_qubits, total_qubits - 1, total_qubits - 8, total_qubits - 3],
    )  # -ve ax[k+1,i,j]
    qc.h( total_qubits - 1)
    # qc.ch(num_total_qubits, total_qubits - 1, ctrl_state="0")

    # # qc.barrier()  # moves to not change AXXX in run 8
    qc.append(
        MCXGate(3, ctrl_state="110"), [num_total_qubits,total_qubits, total_qubits - 1, total_qubits - 2]
    )  # -ve ax[k+1,i,j]
    # qc.append(MCXGate(2, ctrl_state="11"), [total_qubits,  total_qubits-1, total_qubits - 3]) # -ve ax[k+1,i,j]
    # qc.append(MCXGate(2, ctrl_state="11"), [total_qubits,  total_qubits-1, total_qubits - 4]) # -ve ax[k+1,i,j]
    qc.append(
        MCXGate(3, ctrl_state="110"), [num_total_qubits,total_qubits, total_qubits - 1, total_qubits - 7]
    )  # -ve ax[k+1,i,j]

    # qc.barrier()
    qc.append(
        SwapGate().control(2, ctrl_state="00"),
        [num_total_qubits,total_qubits, total_qubits - 1, total_qubits - 3],
    )
    qc.append(
        MCXGate(3, ctrl_state="100"), [num_total_qubits, total_qubits, total_qubits - 1, total_qubits - 8]
    )  # -ve ax[k+1,i,j]
    # qc.ch(num_total_qubits, total_qubits - 1, ctrl_state="0")
    qc.h( total_qubits - 1)
    # qc.barrier()
    # # # qc.append(SwapGate().control(1, ctrl_state="1"), [total_qubits, total_qubits - 1, total_qubits-3])
    # qc.ch(num_total_qubits, total_qubits, ctrl_state="0")
    qc.h( total_qubits)
    qc.barrier()
    
    qc.append(
        SwapGate().control(5, ctrl_state="00000"),
        [   num_total_qubits,
            total_qubits,
            total_qubits - 1,
            total_qubits - 2,
            total_qubits - 3,
            
            total_qubits - 4,
            total_qubits - 6,
        ],
    )
    qc.append(
        SwapGate().control(6, ctrl_state="000000"),
        [   num_total_qubits,
            total_qubits,
            total_qubits - 1,
            total_qubits - 2,
            total_qubits - 3,
            total_qubits - 4,
            total_qubits - 5,
            total_qubits - 7,
        ],
    )

    return qc


def summations(total_qubits, num_qubits):

    num_total_qubits = total_qubits + 1
    qc = QuantumCircuit(num_total_qubits)

    qc.swap(total_qubits - 8, total_qubits)
    qc.append(
        SwapGate().control(2, ctrl_state="00"),
        [total_qubits, total_qubits - 1, total_qubits - 8, total_qubits - 7],
    )
    qc.h(total_qubits - 2)
    qc.barrier()

    qc.append(
        SwapGate().control(1, ctrl_state="0"),
        [total_qubits, total_qubits - 2, total_qubits - 3],
    )
    # # qc.append(MCXGate(5, ctrl_state="00100"), [total_qubits - i for i in range(0, 5)] + [total_qubits - 5]) # -ve ax[k+1,i,j]
    qc.append(
        SwapGate().control(1, ctrl_state="1"),
        [total_qubits, total_qubits - 2, total_qubits - 3],
    )
    # # qc.append(MCXGate(6, ctrl_state="000001"), [total_qubits - i for i in range(0, 6)] + [total_qubits - 6]) # -ve ax[k+1,i,j]
    qc.h(total_qubits - 2)
    qc.barrier()

    qc.append(
        SwapGate().control(1, ctrl_state="1"),
        [total_qubits, total_qubits - 1, total_qubits - 7],
    )
    angles = [math.acos(1 / 4), math.asin(1 / 4)]
    # # qc.barrier()
    state_ctl2 = ["11"]
    for angle, state in zip(angles, state_ctl2):
        CR = gates.RYGate(2 * angle).control(2, ctrl_state=state)

        qc.append(CR, [num_total_qubits - 1, num_total_qubits - 2] + [num_qubits])
    qc.barrier()

    qc.append(
        MCXGate(2, ctrl_state="10"),
        [total_qubits, total_qubits - 1] + [total_qubits - 7],
    )  # -ve ax[k+1,i,j]
    state_ctl3 = ["10"]
    # qc.append(MCXGate(6, ctrl_state="000000"), [total_qubits - i for i in range(0, 6)] + [total_qubits - 6]) # -ve ax[k+1,i,j]
    for angle, state in zip(angles, state_ctl3):
        CR = gates.RYGate(2 * angle).control(2, ctrl_state=state)

        qc.append(CR, [num_total_qubits - 1, num_total_qubits - 2] + [num_qubits + 1])

    qc.append(
        MCXGate(3, ctrl_state="100"),
        [total_qubits, total_qubits - 1, total_qubits - 8, total_qubits - 3],
    )  # -ve ax[k+1,i,j]
    qc.h(total_qubits - 1)

    qc.barrier()  # moves to not change AXXX in run 8
    qc.append(
        MCXGate(2, ctrl_state="11"), [total_qubits, total_qubits - 1, total_qubits - 2]
    )  # -ve ax[k+1,i,j]
    # qc.append(MCXGate(2, ctrl_state="11"), [total_qubits,  total_qubits-1, total_qubits - 3]) # -ve ax[k+1,i,j]
    # qc.append(MCXGate(2, ctrl_state="11"), [total_qubits,  total_qubits-1, total_qubits - 4]) # -ve ax[k+1,i,j]
    qc.append(
        MCXGate(2, ctrl_state="11"), [total_qubits, total_qubits - 1, total_qubits - 7]
    )  # -ve ax[k+1,i,j]

    qc.barrier()
    qc.append(
        SwapGate().control(1, ctrl_state="0"),
        [total_qubits, total_qubits - 1, total_qubits - 3],
    )
    qc.append(
        MCXGate(2, ctrl_state="10"), [total_qubits, total_qubits - 1, total_qubits - 8]
    )  # -ve ax[k+1,i,j]
    qc.h(total_qubits - 1)
    qc.barrier()
    # # # qc.append(SwapGate().control(1, ctrl_state="1"), [total_qubits, total_qubits - 1, total_qubits-3])
    qc.h(total_qubits)
    qc.barrier()
    qc.append(
        SwapGate().control(4, ctrl_state="0000"),
        [
            total_qubits,
            total_qubits - 1,
            total_qubits - 2,
            total_qubits - 3,
            
            total_qubits - 4,
            total_qubits - 6,
        ],
    )
    qc.append(
        SwapGate().control(5, ctrl_state="00000"),
        [
            total_qubits,
            total_qubits - 1,
            total_qubits - 2,
            total_qubits - 3,
            total_qubits - 4,
            total_qubits - 5,
            total_qubits - 7,
        ],
    )

    return qc


def column_to_row_transform(A):
    """
    A: (nz-1, nx, ny)
    returns: (nz, nx-1, ny)
    """

    nzm1, nx, ny = A.shape
    nz = nzm1 + 1

    result = np.zeros((nz, nx - 1, ny))

    # core transform (NO reshape!)
    result[:, :, :] = A[:, :, :].transpose(2, 0, 1)

    return result


def transform(A):
    """
    A: shape (nz-1, nx, ny)
    returns: shape (nz, nx, nz-1)
    """

    nzm1, nx, ny = A.shape
    nz = nzm1 + 1

    result = np.zeros((nz, nx, nzm1))

    # core operation
    result[:, :, :] = A[:, :, :].transpose(2, 1, 0)

    return result


def zxy_to_yzx(a):

    nz, nx, ny = a.shape
    # print(nz, nx, ny)

    anz, anx, any = ny, nz, nx
    # print(anz, anx, any)

    an = np.zeros((anz, anx, any))

    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                an[j, k, i] = a[k, i, j]

    return an


def zxy_to_xyz(a):

    nz, nx, ny = a.shape
    # print(nz, nx, ny)

    anz, anx, any = nx, ny, nz
    # print(anz, anx, any)

    an = np.zeros((anz, anx, any))

    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                an[i, j, k] = a[k, i, j]

    return an


def get_az_ay_ax(state_vector, nx, ny, nz, path):

    filename = path + f"output_ax_ay_az.txt"
    ids = nx * ny * nz
    shape_sv = len(state_vector[: 3 * ids])

    az_shape = shape_sv // 3

    ay_shape = 2 * shape_sv // 3
    ax_shape = shape_sv

    print(az_shape)
    print(ay_shape)
    print(ax_shape)

    len_a = (nz - 1) * nx * ny

    az = state_vector[:len_a]
    ay = state_vector[az_shape * 2: az_shape *2 + len_a]
    ax = state_vector[az_shape * 4 : az_shape * 4 + len_a]

    az = az.reshape(nz - 1, nx, ny)
    ay = ay.reshape(nz - 1, nx, ny)
    ax = ax.reshape(nz - 1, nx, ny)

    tol = 1e-12
    ax[np.abs(ax) < tol] = 0.0
    ay[np.abs(ay) < tol] = 0.0
    az[np.abs(az) < tol] = 0.0
    axn = column_to_row_transform(ax)
    ayn = transform(ay)

    azn = np.copy(az)

    with open(filename, "w") as f:
        section = f"\n--- ax ---\n"

        print(section)
        f.write(section)
        f.write(str(axn))
        section = f"\n--- ay ---\n"

        print(section)
        f.write(section)
        f.write(str(ayn))
        section = f"\n--- az ---\n"

        print(section)
        f.write(section)
        f.write(str(azn))

    return axn, ayn, azn


def calculate_b_and_plot(ax, ay, az, nx, ny, nz, path):

    np.save(path + "data_ax.npy", ax)
    np.save(path + "data_ay.npy", ay)
    np.save(path + "data_az.npy", az)

    bx = np.zeros((nz - 1, nx, ny - 1), dtype=np.float64)
    by = np.zeros((nz - 1, nx - 1, ny), dtype=np.float64)
    bz = np.zeros((nz, nx - 1, ny - 1), dtype=np.float64)

    Bx = np.zeros((nz - 1, nx - 1, ny - 1), dtype=np.float64)
    By = np.zeros((nz - 1, nx - 1, ny - 1), dtype=np.float64)
    Bz = np.zeros((nz - 1, nx - 1, ny - 1), dtype=np.float64)

    for k in range(nz - 1):
        for i in range(nx - 1):
            for j in range(ny - 1):
                bx[k, i, j] = (
                    +az[k, i, j] - az[k, i, j + 1] + ay[k, i, j] - ay[k + 1, i, j]
                )

    for k in range(nz - 1):
        for i in range(nx - 1):
            for j in range(ny - 1):
                by[k, i, j] = (
                    -az[k, i, j] + az[k, i + 1, j] - ax[k, i, j] + ax[k + 1, i, j]
                )

    for k in range(nz):
        for i in range(nx - 1):
            for j in range(ny - 1):
                bz[k, i, j] = (
                    +ax[k, i, j] + ax[k, i, j + 1] + ay[k, i, j] + ay[k, i + 1, j]
                )

    for k in range(nz - 1):
        for i in range(nx - 1):
            for j in range(ny - 1):
                Bx[k, i, j] = (by[k, i, j] + by[k, i, j + 1]) / 2

    for k in range(nz - 1):
        for i in range(nx - 1):
            for j in range(ny - 1):
                By[k, i, j] = (bx[k, i, j] + bx[k, i + 1, j]) / 2

    for k in range(nz - 1):
        for i in range(ny - 1):
            for j in range(nx - 1):
                Bz[k, i, j] = +(bz[k + 1, i, j] + bz[k, i, j]) / 2

    np.save(path + "data_Bx.npy", Bx)
    np.save(path + "data_By.npy", By)
    np.save(path + "data_Bz.npy", Bz)

    # plt.figure()
    # plt.contourf(az[1,:,:], levels=50, cmap='viridis')
    # plt.title('Smoothed Field Z')
    # plt.colorbar()
    # plt.savefig(path + "az.png")

    # plt.figure()
    # X,Y = np.meshgrid(np.arange(ny-1), np.arange(nx-1))
    # plt.quiver(X, Y, Bx[0,:,:], By[0,:,:])
    # plt.title('Vector Field B')
    # plt.savefig(path + "quiver_B.png")

    # b = np.sqrt(Bx**2 + By**2)
    # plt.figure()
    # plt.contourf(b[1,:,:], levels=50, cmap='plasma')
    # plt.title('Magnitude of B Field')
    # plt.colorbar()
    # plt.savefig(path + "mag_B.png")


def save_statevector(
    final_statevector, statevec_norm, hadamard_copies, idddd, path, nx
):

    results = {}

    scale = statevec_norm * np.sqrt(2) ** hadamard_copies
    block = idddd
    num_blocks = 16  # axxx, ayyy, azzz, jxxx, jyyy, jzzz

    offset = 0
    resultss = []
    for run in range(32):

        results[run] = {}
        names = [
            "axz",
            "ayz",
            "azz",
            "jz",
            "azy",
            "axy",
            "ayy",
            "jy",
            "ayx",
            "azx",
            "axx",
            "jx",
            "no_ned1",
            "no_ned2",
            "no_ned3",
            "no_ned4",
        ]

        for k, name in enumerate(names):
            start = offset + k * block
            end = start + block
            results[run][name] = final_statevector[start:end].real * scale
            resultss.append(results)

        offset += num_blocks * block

    filename = path + f"output_beta_incidence_1.txt"
    runs_to_print = range(16)
    fields = [
        "axz",
        "ayz",
        "azz",
        "jz",
        "azy",
        "axy",
        "ayy",
        "jy",
        "ayx",
        "azx",
        "axx",
        "jx",
        "no_ned1",
        "no_ned2",
        "no_ned3",
        "no_ned4",
    ]

    block = idddd

    num_blocks = 16
    n_qubits = int(np.log2(len(final_statevector)))
    data_to_save = {}
    if not os.path.exists(filename):
        open(filename, "w").close()

    with open(filename, "w") as f:  # append mode

        for run in runs_to_print:

            offset = run * num_blocks * block

            header = f"\n{'='*40} RUN {run} {'='*40}\n"
            # print(header)
            f.write(header)

            for k, field in enumerate(fields):

                start = offset + k * block
                binary_start = format(start, f"0{n_qubits}b")

                section = (
                    f"\n--- {field.upper()} ---\n"
                    f"Start index: {start}\n"
                    f"Binary start: {binary_start[0] } "
                    f"{binary_start[1:4] } "
                    f"{binary_start[4:8] } "
                    f"{binary_start[8:]}\n"
                )

                # print(section)
                f.write(section)
                # if field == "ayz" or field == "axz" or field == "azz":
                reshaped = results[run][field].reshape((nx, nx, nx)).round(3)
                f.write(str(reshaped))
                # Save 3D array slice by slice
                # for slice_ in reshaped:
                #     np.savetxt(f, slice_, fmt="%.1f")
                #     f.write("\n")

                f.write("\n")  # spacing between fields
                if (k + 1) % 4 == 0:
                    f.write("++++++++++++++++++++++++++++++++++++")

                key = f"run{run}_{field}"
                data_to_save[key] = reshaped

    # np.savez(path + f"results_data_relaxation_incidence.npz", **data_to_save)


def padding_as(ax, ay, az):
    ax = np.pad(ax, pad_width=((0, 0), (0, 1), (0, 0)), mode="constant")
    ay = np.pad(ay, pad_width=((0, 0), (0, 0), (0, 1)), mode="constant")
    az = np.pad(az, pad_width=((0, 1), (0, 0), (0, 0)), mode="constant")

    return ax, ay, az


def flatten_as(ax, ay, az):
    ax_flat = ax.flatten()
    ay_flat = ay.flatten()
    az_flat = az.flatten()

    return ax_flat, ay_flat, az_flat

def apply_boundary_conditions(ax, ay, az):
    # ax boundaries
    ax[0, :, :] = 0
    ax[-1, :, :] = 0
    ax[:, :, 0] = 0
    ax[:, :, -1] = 0

    # ay boundaries
    ay[0, :, :] = 0
    ay[-1, :, :] = 0
    ay[:, 0, :] = 0
    ay[:, -1, :] = 0

    # az boundaries
    az[:, 0, :] = 0
    az[:, -1, :] = 0
    az[:, :, 0] = 0
    az[:, :, -1] = 0

    return ax, ay, az

def check_convergence(ax, ay, az, axn, ayn, azn):
    norm2_x = np.linalg.norm(axn - ax)**2
    norm2_y = np.linalg.norm(ayn - ay)**2
    norm2_z = np.linalg.norm(azn - az)**2

    print("norm of ax for converg :", norm2_x)
    print("norm of ay for converg :", norm2_y)
    print("norm of az for converg :", norm2_z)

    return norm2_x, norm2_y, norm2_z


def save_statevector_with_pi(
    final_statevector, statevec_norm, hadamard_copies, idddd, path, nx
):

    results = {}

    scale = statevec_norm * np.sqrt(2) ** hadamard_copies
    block = idddd
    num_blocks = 8  # axxx, ayyy, azzz, jxxx, jyyy, jzzz

    offset = 0
    # resultss = []
    
    filename = path + f"output_beta_incidence_1.txt"
    runs_to_print = range(128)
    fields = [
            "axz",
            "ayz",
            "azz",
            "jz",
            "jy",
            "jx",
            "no_1",
            "no_2",
        ]

    block = idddd

    
    n_qubits = int(np.log2(len(final_statevector)))
    data_to_save = {}
    if not os.path.exists(filename):
        open(filename, "w").close()

    with open(filename, "w") as f:  # append mode

        for run in runs_to_print:

            offset = run * num_blocks * block

            header = f"\n{'='*15} RUN {run} {'='*15}\n"
            # print(header)
            f.write(header)
            results[run] = {}

            for k, field in enumerate(fields):

                start = offset + k * block
                binary_start = format(start, f"0{n_qubits}b")
                end = start + block
                results[run][field] = final_statevector[start:end].real * scale
                # resultss.append(results)

                section = (
                    f"\n--- {field.upper()} ---\n"
                    f"Start index: {start}\n"
                    f"Binary start: {binary_start[0:2] } "
                    f"{binary_start[2:5] } "
                    f"{binary_start[5:7] } "
                    f"{binary_start[7:10] } "
                    f"{binary_start[10:]}\n"
                )

                # print(section)
                f.write(section)
                # if field == "ayz" or field == "axz" or field == "azz":
                reshaped = results[run][field].reshape((nx, nx, nx)).round(3)
                f.write(str(reshaped))
                # Save 3D array slice by slice
                # for slice_ in reshaped:
                #     np.savetxt(f, slice_, fmt="%.1f")
                #     f.write("\n")

                f.write("\n")  # spacing between fields
                if (k + 1) % 4 == 0:
                    f.write("\n++++++++++++++++++++++++++++++++++++")

                key = f"run{run}_{field}"
                data_to_save[key] = reshaped

    # np.savez(path + f"results_data_relaxation_incidence.npz", **data_to_save)

def quantum_pi_rotations(num_qubits_total, numq_dir1, numq_dir2, numq_dir3, ctrls, ctrlstate):

    qc = QuantumCircuit(num_qubits_total)
    swapgate_pi = SwapGate().control(len(ctrlstate), ctrl_state=ctrlstate)
    for i in range(numq_dir2):
        qc.append(
            swapgate_pi,
            ctrls + [numq_dir1 + i, numq_dir3 + i]
        )

    for i in range(numq_dir2):
        qc.append(
            swapgate_pi,
            ctrls + [numq_dir1 + i, numq_dir2 + i]
        )

    qc.append(swapgate_pi, ctrls+[numq_dir3 + numq_dir2, numq_dir3 + numq_dir2 + 1])
    qc.append(MCXGate(len(ctrlstate)+1, ctrl_state="0"+ctrlstate), ctrls +[ctrls[0] - 2] + [numq_dir3 + numq_dir2])

    return qc