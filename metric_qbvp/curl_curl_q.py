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
    HGate,
    StatePreparation,
)

import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import argparse

from curl_curl_functions import (
    streaming_axz,
    streaming_ayz,
    streaming_azz,
    save_statevector,
    summations,
    relaxation_param,
    calculate_b_and_plot,
    zxy_to_xyz,
    zxy_to_yzx,
    get_az_ay_ax
)

np.set_printoptions(threshold=sys.maxsize, linewidth=200)
path = "/home/tmvshinde/E-Folder/PHD/Codes/curl_curl/"
# path = "/scratch/project_462001154/scratch_curl_curl/"

parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, help="Number of grid points in one direction")
parser.add_argument("-t", type=int, help="Number of timesteps")
args = parser.parse_args()
# print("m =", args.m)
# print("t =", args.t)

num_grid_dir_one = args.m
nx, ny, nz = num_grid_dir_one, num_grid_dir_one, num_grid_dir_one
ax = np.zeros((nz, nx - 1, ny), dtype=np.float64)
ay = np.zeros((nz, nx, ny - 1), dtype=np.float64)
az = np.zeros((nz - 1, nx, ny), dtype=np.float64)

# ax = np.arange(1, nz * (nx-1) * ny + 1).reshape(nz, nx-1, ny)
# ay = np.arange(1, nz * nx * (ny-1) + 1).reshape(nz, nx, ny-1)
# az = np.arange(1, (nz-1) * nx * ny + 1).reshape(nz-1, nx, ny)
# # print(ax)

jx = np.zeros((nx - 1, ny, nz), dtype=np.float64)
jy = np.zeros((ny - 1, nz, nx), dtype=np.float64)
jz = np.zeros((nz - 1, nx, ny), dtype=np.float64)

jz[:, nx // 2, ny // 2] = 1


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


jx = np.pad(jx, pad_width=((0, 1), (0, 0), (0, 0)), mode="constant")
jy = np.pad(jy, pad_width=((0, 1), (0, 0), (0, 0)), mode="constant")
jz = np.pad(jz, pad_width=((0, 1), (0, 0), (0, 0)), mode="constant")


Beta = 0.8
beta = 0.8
timesteps = args.t


jx_flat = jx.flatten()
jy_flat = jy.flatten()
jz_flat = jz.flatten()

for t in range(timesteps):

    print(f"============Iteration {t}========= ")

    axz = ax
    ayz = ay
    azz = az

    axy = zxy_to_yzx(az)
    ayy = zxy_to_yzx(ax)
    azy = zxy_to_yzx(ay)

    axx = zxy_to_xyz(ay)
    ayx = zxy_to_xyz(az)
    azx = zxy_to_xyz(ax)

    ax_new = np.zeros((nx - 1, ny, nz), dtype=np.float64)
    ay_new = np.zeros((ny - 1, nz, nx), dtype=np.float64)
    az_new = np.zeros((nz - 1, nx, ny), dtype=np.float64)

    axz, ayz, azz = padding_as(axz, ayz, azz)
    axy, ayy, azy = padding_as(axy, ayy, azy)
    axx, ayx, azx = padding_as(axx, ayx, azx)

    axz_flat, ayz_flat, azz_flat = flatten_as(axz, ayz, azz)
    axy_flat, ayy_flat, azy_flat = flatten_as(axy, ayy, azy)
    axx_flat, ayx_flat, azx_flat = flatten_as(axx, ayx, azx)
    # print(axz_flat)
    num_q_ax = np.log2(axz_flat.shape[0]).astype(int)
    print("num_q_ax: ", num_q_ax)

    state_vec = np.concatenate(
        (
            axz_flat,
            ayz_flat,
            azz_flat,
            jz_flat,
            axy_flat,
            ayy_flat,
            azy_flat,
            jy_flat,
            axx_flat,
            ayx_flat,
            azx_flat,
            jx_flat,
        )
    )
    # print(state_vec)
    state_vec = np.pad(
        state_vec,
        (0, 2 ** math.ceil(np.log2(state_vec.shape[0])) - state_vec.shape[0]),
        mode="constant",
    )
    print("state_vec shape: ", state_vec.shape[0])

    # calculating the number of qubits needed to represent the ax, ay, az, jx, jy, jz arrays as a quantum state vector
    num_qubits = np.log2(state_vec.shape[0]).astype(int)
    # print(num_qubits.dtype)
    # 4 qubits needed for the 12 hadamard copies
    hadamard_copies = 3

    # total number of qubits needed
    total_qubits = num_qubits + hadamard_copies
    num_total_qubits = total_qubits + 1
    # print(total_qubits.dtype)

    print("Number of qubits needed: ", num_qubits)

    print("num of states ", 2**num_qubits)

    # print("idle states", 2**num_qubits - state_vec.shape[0])

    num_q_xyz = num_qubits - 4
    numqz = 0
    numqx = num_q_xyz - 2 * (num_q_xyz // 3)
    numqy = num_q_xyz - numqx

    print("numqz: ", numqz, " numqx: ", numqx, "numqy: ", numqy)

    qc = QuantumCircuit(num_total_qubits)

    statevec_norm = np.linalg.norm(state_vec)
    print("Norm of the state vector before normalization: ", statevec_norm)
    state_vec = state_vec / statevec_norm

    # print(state_vec * statevec_norm)

    qubits_for_init = np.arange(0, num_qubits)

    # print("Qubits used for state initialization: ", qubits_for_init)

    # prep = StatePreparation(state_vec)
    # qc.append(prep, range(0, num_qubits))
    qc.initialize(state_vec, range(0, num_qubits), normalize=False)

    qc.h(range(num_qubits, total_qubits))
    hs = 3

    # # az
    ctrls_az = "01"
    ctrls_ax = "00"
    ctrls_ay = "10"
    num_copies = 5
    ctrls_copies = []
    num_h_qubits = total_qubits - num_qubits
    for i in range(num_copies):
        bin_copies = bin(i)[2:].zfill(num_h_qubits)
        ctrls_copies.append(bin_copies)

    # print(ctrls_copies)
    qcz = streaming_azz(
        total_qubits, numqx, numqz, ctrls_az, ctrls_copies
    )  # az[k,i+1,j], az[k,i-1,j], az[k,i,j+1], az[k,i,j-1]
    qc.append(qcz.to_instruction(label="Z"), range(total_qubits + 1))
    # block_gate = streaming_azz(qc, total_qubits, numqx, numqz, ctrls_az, ctrls_copies).to_instruction()
    qc.barrier()

    # # # ax
    # qc = streaming_axz(qc, total_qubits, numqx, numqy, ctrls_ax, ctrls_copies) # ax[k+1,i,j], ax[k,i-1,j]
    qcx = streaming_axz(
        total_qubits, numqx, numqy, ctrls_ax, ctrls_copies
    )  # ax[k+1,i,j], ax[k,i-1,j]qc = test_streaming_axz(qc, total_qubits-2, numqx, numqy, ctrls_ax, ctrls_copies) # ax[k+1,i,j], ax[k,i-1,j]
    qc.append(qcx.to_instruction(label="X"), range(total_qubits + 1))
    qc.barrier()
    # # # ay
    # qc = streaming_ayz(qc, total_qubits, numqy, numqz, ctrls_ay, ctrls_copies) # ay[k,i,j-1], ay[k+1,i,j-1]
    qcy = streaming_ayz(
        total_qubits, numqy, numqz, ctrls_ay, ctrls_copies
    )  # ay[k,i,j-1], ay[k+1,i,j-1]
    qc.append(qcy.to_instruction(label="Y"), range(total_qubits + 1))
    qc.barrier()
    # this is relaxation parameter

    angle_p = [
        math.acos(beta / 4),
        math.asin(-beta / 4),
        math.acos(1 - beta),
        math.asin(1 - beta),
    ]

    state_ctl = [
        "0",
        "1",
    ]
    ctrls = ["000", "100", "001", "101", "010"]
    ctrls = ["00", "01", "10", "11"]

    qcrp = relaxation_param(ctrls, angle_p, num_total_qubits)
    qc.append(qcrp.to_instruction(label="Relaxation"), range(num_total_qubits))
    qc.barrier()

    hadamardss = 5
    change_file_name = "tot_enc_h3_sz_relax_sum1_sum2_sum3_sum4"

    qcsum = summations(total_qubits, num_qubits)
    qc.append(qcsum.to_instruction(label="Summations"), range(num_total_qubits))
    qc.barrier()

    # qc.draw('mpl', fold=30, filename=path + "curclq_beta_total_iter_8.png")
    qc.save_statevector()

    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit).result()
    final_statevector = np.asarray(result.get_statevector())

    hadamard_copie = hs + (hadamardss)

    final_statevector_ = (
        final_statevector.real * statevec_norm * np.sqrt(2) ** (hadamard_copie)
    )
    idddd = nx * ny * nz
    # save_statevector(final_statevector, statevec_norm, hadamard_copie, idddd, path, nx)

    axn, ayn, azn = get_az_ay_ax(final_statevector_, nx, ny, nz, path)

    axn[0, :, :] = 0
    axn[:, :, 0] = 0
    axn[:, :, -1] = 0
    axn[-1, :, :] = 0

    ayn[0, :, :] = 0
    ayn[:, 0, :] = 0
    ayn[:, -1, :] = 0
    ayn[-1, :, :] = 0

    azn[:, 0, :] = 0
    azn[:, -1, :] = 0
    azn[:, :, 0] = 0
    azn[:, :, -1] = 0

    norm_ax = axn - ax
    norm_convergx = np.linalg.norm(norm_ax)
    print("norm of ax for converg : ", norm_convergx**2)

    norm_ay = ayn - ay
    norm_convergy = np.linalg.norm(norm_ay)
    print("norm of ay for converg : ", norm_convergy**2)

    norm_az = azn - az
    norm_converg = np.linalg.norm(norm_az)
    print("norm of az for converg : ", norm_converg**2)

    if norm_converg**2 < 0.0000001:
        break

    az = np.copy(azn)
    ax = np.copy(axn)
    ay = np.copy(ayn)

    # boundary conditions
    ax[0, :, :] = 0
    ax[:, :, 0] = 0
    ax[:, :, -1] = 0
    ay[0, :, :] = 0
    ay[:, 0, :] = 0
    ay[:, -1, :] = 0
    ax[-1, :, :] = 0
    ay[-1, :, :] = 0
    az[:, 0, :] = 0
    az[:, -1, :] = 0
    az[:, :, 0] = 0
    az[:, :, -1] = 0

# print(az.round(3))

calculate_b_and_plot(ax, ay, az, nx, ny, nz, path)
