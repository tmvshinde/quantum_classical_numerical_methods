"""
One place for saving quantum circuits.

    from circuit_saving import save_circuit, save_circuits
    save_circuit(qc, "circuits/my_test", "my_circuit")          # -> my_circuit.svg (+ .qpy)
    save_circuits({"blocks": blk, "gates": det}, "circuits/my_test")

Every diagram is written as SVG; a .qpy copy is kept so the circuit can be reloaded exactly.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import qpy


def save_circuit(qc, folder, name, fold=50, scale=0.7, qpy_copy=True):
    """Save the circuit diagram as <folder>/<name>.svg (and <name>.qpy)."""
    os.makedirs(folder, exist_ok=True)
    fig = qc.draw("mpl", fold=fold, scale=scale, style={"name": "iqp"})
    fig.savefig(os.path.join(folder, f"{name}.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    if qpy_copy:
        with open(os.path.join(folder, f"{name}.qpy"), "wb") as fh:
            qpy.dump(qc, fh)
    print(f"saved {folder}/{name}.svg" + (" / .qpy" if qpy_copy else ""))


def save_circuits(circuits, folder, **kwargs):
    """Save several circuits: circuits = {name: QuantumCircuit or (QuantumCircuit, draw-kwargs)}."""
    for name, item in circuits.items():
        qc, opts = item if isinstance(item, tuple) else (item, {})
        save_circuit(qc, folder, name, **{**kwargs, **opts})
