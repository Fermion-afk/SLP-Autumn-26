"""LiH / STO-3G SS-COMPASS-VQE runner without frozen-core approximation."""

from datetime import datetime
from pathlib import Path
import contextlib
import sys
import traceback
from typing import cast,Any

import ss_compass_vqe as scv

class NoFreezeCoreTransformer:
    """Identity replacement for Qiskit Nature's FreezeCoreTransformer."""

    def __init__(self, *args, **kwargs):
        pass

    def transform(self, es_problem):
        return es_problem

# Override only inside this runner; ss_compass_vqe.py remains unchanged.
scv.FreezeCoreTransformer = NoFreezeCoreTransformer

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def write_screening_sheet(ws, records, state_count, kind):
    state_columns = [f"Delta_E_state_{i}" for i in range(state_count)]
    if kind == "T2":
        headers = ["Operator_ID", "Excitation", *state_columns, "Weighted_Sum",
                   "Threshold", "Selected", "Theta_T2"]
    else:
        headers = ["Candidate_ID", "Parent_T2_ID", "Type", "Operator_ID", "Excitation",
                   *state_columns, "Weighted_Sum", "Threshold", "Selected",
                   "Theta_T2_Initial", "Theta_T2_Final", "Theta_S"]

    ws.append(headers)
    for item in records:
        deltas = item.get("delta_energies", [])
        if kind == "T2":
            row = [item["operator_id"], item["excitation"], *deltas,
                   item["weighted_sum"], item["threshold"], item["selected"],
                   item["theta_t2"]]
        else:
            row = [item["candidate_id"], item["parent_t2_id"], item["type"],
                   item["operator_id"], item["excitation"], *deltas,
                   item["weighted_sum"], item["threshold"], item["selected"],
                   item["theta_t2_initial"], item["theta_t2_final"], item["theta_s"]]
        ws.append(row)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

def main():
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    bond_length = 1.5949
    basis_set = "sto-3g"
    coordinates = f"Li 0.0 0.0 0.0; H 0.0 0.0 {bond_length}"
    n_states = 3
    weights = [3.0, 2.0, 1.0]
    thresholds = [1e-5, 5e-6]

    timestamp = datetime.now().astimezone().strftime("%d.%m.%y#%H.%M")
    stem = f"LiH_no_frozen_core#{timestamp}"
    log_path = Path(f"{stem}.log")
    excel_path = Path(f"{stem}.xlsx")

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee = Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(cast(Any,tee)), contextlib.redirect_stderr(cast(Any,tee)):
            print("=" * 72)
            print("LiH SS-COMPASS-VQE | STO-3G | NO FROZEN CORE")
            print(f"Bond length: {bond_length} Angstrom")
            print(f"Reference states: {n_states}")
            print(f"Weights: {weights}")
            print(f"Screening thresholds: {thresholds}")
            print("Frozen-core transformer: disabled by runner-level override")
            print("=" * 72, flush=True)

            result = scv.ss_compass_vqe(
                coordinates=coordinates, d_eq=bond_length, basis_set=basis_set,
                no_init_states=n_states, weights=weights, charge=0, spin=0,
                method="cis", threshold=thresholds,
            )
            if result is None:
                raise RuntimeError("ss_compass_vqe returned None.")

            wb = Workbook()
            summary = wb.active
            if summary is None:
                            raise ValueError("summary is None")
            summary.title = "Run Summary"
            for row in [
                ("Molecule", "LiH"), ("Geometry (Angstrom)", bond_length),
                ("Basis set", basis_set), ("Frozen core", "None"),
                ("Charge", 0), ("Spin", 0), ("Initial states", n_states),
                ("Weights", str(weights)), ("T2 threshold", thresholds[0]),
                ("S threshold", thresholds[1]),
                ("Number of final parameters", result["n_params"]),
            ]:
                summary.append(row)

            summary.append([])
            summary.append(["State", "SS-COMPASS Energy (Ha)",
                            "Exact Fixed-N Energy (Ha)", "Error (Ha)",
                            "Initial Energy (Ha)"])
            for i, energy in enumerate(result["energies"]):
                exact = result["exact_energies"][i] if i < len(result["exact_energies"]) else None
                initial = result["initial_energies"][i] if i < len(result["initial_energies"]) else None
                summary.append([i, float(energy), float(exact) if exact is not None else None,
                                float(energy - exact) if exact is not None else None,
                                float(initial) if initial is not None else None])

            write_screening_sheet(wb.create_sheet("T2 Screening"),
                                  result.get("t2_screening", []), n_states, "T2")
            write_screening_sheet(wb.create_sheet("S Screening"),
                                  result.get("s_screening", []), n_states, "S")

            for ws in wb.worksheets:
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                for column in ws.columns:
                    letter = get_column_letter(cast(Any,column[0].column))
                    width = min(max(len(str(cell.value or "")) for cell in column) + 2, 42)
                    ws.column_dimensions[letter].width = max(ws.column_dimensions[letter].width or 0, width)

            wb.save(excel_path)
            print(f"\nExcel report saved: {excel_path.resolve()}")
            print(f"Log saved: {log_path.resolve()}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
