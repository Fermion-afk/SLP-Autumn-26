"""LiH / STO-3G SS-COMPASS-VQE runner without frozen-core approximation."""

from datetime import datetime
from pathlib import Path
import contextlib
import multiprocessing as mp
import sys
import traceback
from typing import Any, cast

import ss_compass_vqe as scv


class NoFreezeCoreTransformer:
    """Identity replacement for Qiskit Nature's FreezeCoreTransformer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def transform(self, es_problem: Any) -> Any:
        return es_problem


# Keep the production module unchanged; disable frozen-core reduction only here.
scv.FreezeCoreTransformer = NoFreezeCoreTransformer


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def main() -> None:
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
        with contextlib.redirect_stdout(cast(Any, tee)), contextlib.redirect_stderr(cast(Any, tee)):
            print("=" * 72)
            print("LiH SS-COMPASS-VQE | STO-3G | NO FROZEN CORE")
            print(f"Bond length: {bond_length} Angstrom")
            print(f"Reference states: {n_states}")
            print(f"Weights: {weights}")
            print(f"Screening thresholds: {thresholds}")
            print("Frozen-core transformer: disabled by runner-level override")
            print("=" * 72, flush=True)

            result = scv.ss_compass_vqe(
                coordinates=coordinates,
                d_eq=bond_length,
                basis_set=basis_set,
                no_init_states=n_states,
                weights=weights,
                charge=0,
                spin=0,
                method="cis",
                threshold=thresholds,
            )
            if result is None:
                raise RuntimeError("ss_compass_vqe returned None.")

            workbook = Workbook()
            summary = workbook.active
            if summary is None:
                raise RuntimeError("Workbook did not provide an active worksheet.")
            summary.title = "Run Summary"
            for row in [
                ("Molecule", "LiH"),
                ("Geometry (Angstrom)", bond_length),
                ("Basis set", basis_set),
                ("Frozen core", "None"),
                ("Charge", 0),
                ("Spin", 0),
                ("Initial states", n_states),
                ("Weights", str(weights)),
                ("T2 threshold", thresholds[0]),
                ("S threshold", thresholds[1]),
                ("Number of final parameters", result["n_params"]),
            ]:
                summary.append(row)

            summary.append([])
            summary.append(
                [
                    "State",
                    "SS-COMPASS Energy (Ha)",
                    "Exact Fixed-N Energy (Ha)",
                    "Error (Ha)",
                    "Initial Energy (Ha)",
                ]
            )
            for i, energy in enumerate(result["energies"]):
                exact = result["exact_energies"][i] if i < len(result["exact_energies"]) else None
                initial = result["initial_energies"][i] if i < len(result["initial_energies"]) else None
                summary.append(
                    [
                        i,
                        float(energy),
                        float(exact) if exact is not None else None,
                        float(energy - exact) if exact is not None else None,
                        float(initial) if initial is not None else None,
                    ]
                )

            ansatz = workbook.create_sheet("Final Ansatz")
            ansatz.append(["Index", "Type", "Excitation", "Theta"])
            for index, item in enumerate(result.get("op_info", []), 1):
                ansatz.append([index, item["type"], str(item["excitation"]), item["theta"]])

            for worksheet in workbook.worksheets:
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)
                for column in worksheet.columns:
                    letter = get_column_letter(cast(Any, column[0].column))
                    width = min(max(len(str(cell.value or "")) for cell in column) + 2, 42)
                    worksheet.column_dimensions[letter].width = max(
                        worksheet.column_dimensions[letter].width or 0, width
                    )

            workbook.save(excel_path)
            print(f"\nExcel report saved: {excel_path.resolve()}")
            print(f"Log saved: {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
