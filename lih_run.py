from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import sys
import traceback

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from ss_compass_vqe import ss_compass_vqe

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str, /) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

def write_screening_excel(result, filename, run_info):
    wb = Workbook()
    summary = wb.active
    if summary is None:
        raise RuntimeError("Workbook has no active worksheet")
    summary.title = "Run Summary"
    rows = [
        ("Molecule", "LiH"),
        ("Geometry (Å)", run_info["bond_length"]),
        ("Geometry source", "Experimental Re, LiH X 1Σ+; DOI: 10.1088/1674-1056/ad20de, Table 1"),
        ("Basis set", run_info["basis"]),
        ("Charge", run_info["charge"]),
        ("Spin", run_info["spin"]),
        ("Reference states", run_info["n_states"]),
        ("Weights", str(run_info["weights"])),
        ("T2 threshold", run_info["thresholds"][0]),
        ("S threshold", run_info["thresholds"][1]),
        ("Initial state energies (Ha)", str(result["initial_energies"].tolist())),
        ("Final state energies (Ha)", str(result["energies"].tolist())),
        ("Exact fixed-N energies (Ha)", str(result["exact_energies"].tolist())),
        ("Final ansatz parameters", result["n_params"]),
    ]
    for row in rows:
        summary.append(row)

    n_states = run_info["n_states"]
    delta_headers = [f"Delta E State {i+1} (Ha)" for i in range(n_states)]

    t2 = wb.create_sheet("T2 Screening")
    t2.append(["Operator ID", "Excitation", *delta_headers, "Weighted Sum (Ha)", "Threshold", "Selected", "Optimized theta_T2"])
    for item in result["t2_screening"]:
        t2.append([item["operator_id"], item["excitation"], *item["delta_energies"], item["weighted_sum"],
                   item["threshold"], item["selected"], item["theta_t2"]])

    s = wb.create_sheet("S Screening")
    s.append(["Candidate ID", "Parent T2 ID", "S Type", "S Operator ID", "Excitation",
              *delta_headers, "Weighted Sum (Ha)", "Threshold", "Selected",
              "Initial theta_T2", "Final theta_T2", "Optimized theta_S"])
    for item in result["s_screening"]:
        s.append([item["candidate_id"], item["parent_t2_id"], item["type"], item["operator_id"],
                  item["excitation"], *item["delta_energies"], item["weighted_sum"], item["threshold"],
                  item["selected"], item["theta_t2_initial"], item["theta_t2_final"], item["theta_s"]])

    for ws in wb.worksheets:
        ws.freeze_panes = "A2" if ws.title != "Run Summary" else "A1"
        if ws.title != "Run Summary":
            ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="244062")
            cell.alignment = Alignment(horizontal="center")
        for col in ws.columns:
            width = min(max(max(len(str(c.value or "")) for c in col) + 2, 12), 42)
            column_index = col[0].column
            if isinstance(column_index, int):
                ws.column_dimensions[get_column_letter(column_index)].width = width
    wb.save(filename)


def main():
    # Experimental equilibrium distance for LiH X 1Sigma+: Re = 1.5957 Å.
    # Source: Phys. Chem. Chem. Phys. (2023), DOI: 10.1088/1674-1056/ad20de, Table 1.
    bond_length = 1.5957
    coordinates = f"Li 0.0 0.0 0.0; H 0.0 0.0 {bond_length}"
    basis = "sto-3g"
    charge, spin = 0, 0

    n_states = 3
    weights = [50.0, 25.0, 1.0]
    thresholds = [1e-5, 5e-6]

    timestamp = datetime.now().strftime("%d.%m.%y#%H.%M")
    run_name = f"LiH#{timestamp}"
    log_file, excel_file = f"{run_name}.log", f"{run_name}.xlsx"
    run_info = {"bond_length": bond_length, "basis": basis, "charge": charge, "spin": spin,
                "n_states": n_states, "weights": weights, "thresholds": thresholds}

    with open(log_file, "w", buffering=1) as log:
        tee = Tee(sys.stdout, log)
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                print(f"Run: {run_name}")
                print(f"Geometry: {coordinates}")
                print(f"Basis: {basis}")
                print(f"Reference states: {n_states}; weights: {weights}")
                print(f"Thresholds [T2, S]: {thresholds}")
                print("-" * 70)

                result = ss_compass_vqe(coordinates=coordinates, d_eq=bond_length, basis_set=basis,
                    no_init_states=n_states, weights=weights, charge=charge, spin=spin,
                    method="Single det", threshold=thresholds)

                write_screening_excel(result, excel_file, run_info)
                print(f"\nExcel screening report saved: {excel_file}")
                print(f"Log saved: {log_file}")
        except Exception:
            traceback.print_exc(file=tee)
            raise

if __name__ == "__main__":
    main()
