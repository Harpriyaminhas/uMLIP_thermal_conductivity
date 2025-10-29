import os
import datetime
import warnings
import traceback
from typing import Literal, Any
from collections.abc import Callable
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

from ase.constraints import FixSymmetry
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from ase import Atoms
from ase.io import read

from k_srme import aseatoms2str, two_stage_relax, ID, NO_TILT_MASK
from k_srme.utils import symm_name_map, get_spacegroup_number, check_imaginary_freqs
from k_srme.conductivity import (
    init_phono3py,
    get_fc2_and_freqs,
    get_fc3,
    calculate_conductivity,
)

#import torch
from mattersim.forcefield import MatterSimCalculator

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

# ---------------- CONFIG ----------------
model_name = "MatterSim-V1"
calc = MatterSimCalculator(device="cpu", load_path="best_model.pth")
checkpoint = "best_model.pth"
suffix = "1M"

ase_optimizer: Literal["FIRE", "LBFGS"] = "FIRE"
ase_filter: Literal["frechet", "exp"] = "frechet"
if_two_stage_relax = True
max_steps = 300
force_max = 1e-4

symprec = 1e-5
enforce_relax_symm = True
conductivity_broken_symm = True  # Always attempt LTC
prog_bar = True
save_forces = True

task_type = "LTC"
job_name = f"{model_name}-POSCARs-{task_type}-{ase_optimizer}{'_2SR' if if_two_stage_relax else ''}_force{force_max}_sym{symprec}"
module_dir = os.path.dirname(__file__)
out_dir = f"{module_dir}/{datetime.datetime.now().strftime('%Y-%m-%d')}-{job_name}"
os.makedirs(out_dir, exist_ok=True)

out_path = f"{out_dir}/conductivity_results.json.gz"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------- Read POSCARs ----------------
poscar_dir = os.path.join(module_dir, "POSCARs_mp_conventional")
poscar_files = sorted([os.path.join(poscar_dir, f) for f in os.listdir(poscar_dir) if f.startswith("POSCAR-mp-")])

atoms_list: list[Atoms] = []
for file in poscar_files:
    atoms = read(file, format="vasp")
    atoms.info[ID] = os.path.basename(file).replace("POSCAR-", "").strip()
    atoms.info["name"] = os.path.basename(file)
    atoms.info["symm.no"] = get_spacegroup_number(atoms, symprec=symprec)
    atoms_list.append(atoms)

print(f"\nJob {job_name} started {timestamp}")
print(f"Found {len(atoms_list)} POSCARs in {poscar_dir}")

# ---------------- Relax + conductivity ----------------
filter_cls: Callable[[Atoms], Atoms] = {"frechet": FrechetCellFilter, "exp": ExpCellFilter}[ase_filter]
optim_cls: Callable[..., Optimizer] = {"FIRE": FIRE, "LBFGS": LBFGS}[ase_optimizer]

force_results: dict[str, dict[str, Any]] = {}
kappa_results: dict[str, dict[str, Any]] = {}

tqdm_bar = tqdm(atoms_list, desc="Conductivity calculation: ", disable=not prog_bar)

for atoms in tqdm_bar:
    mat_id = atoms.info[ID]
    init_info = deepcopy(atoms.info)
    mat_name = atoms.info["name"]
    mat_desc = f"{mat_name}-{symm_name_map.get(atoms.info['symm.no'], 'UNK')}"
    info_dict = {
        "desc": mat_desc,
        "name": mat_name,
        "initial_space_group_number": atoms.info["symm.no"],
        "errors": [],
        "error_traceback": [],
    }

    tqdm_bar.set_postfix_str(mat_desc, refresh=True)

    # ---------- Relaxation ----------
    try:
        atoms.calc = calc
        if max_steps > 0:
            if not if_two_stage_relax:
                filtered_atoms = filter_cls(atoms, mask=NO_TILT_MASK) if enforce_relax_symm else filter_cls(atoms)
                if enforce_relax_symm:
                    atoms.set_constraint(FixSymmetry(atoms))

                optimizer = optim_cls(filtered_atoms, logfile=f"{out_dir}/relax_{mat_id}.log")
                optimizer.run(fmax=force_max, steps=max_steps)

                reached_max_steps = optimizer.step == max_steps
                if reached_max_steps:
                    print(f"⚠️ {mat_desc} ({mat_id}) reached max steps {max_steps}")

                max_stress = atoms.get_stress().reshape((2, 3), order="C").max(axis=1)
                atoms.calc = None
                atoms.constraints = None
                atoms.info = init_info | atoms.info
                symm_no = get_spacegroup_number(atoms, symprec=symprec)

                relax_dict = {
                    "structure": aseatoms2str(atoms),
                    "max_stress": max_stress.tolist(),
                    "reached_max_steps": reached_max_steps,
                    "relaxed_space_group_number": symm_no,
                    "broken_symmetry": symm_no != init_info["symm.no"],
                }
            else:
                atoms, relax_dict = two_stage_relax(
                    atoms,
                    fmax_stage1=force_max,
                    fmax_stage2=force_max,
                    steps_stage1=max_steps,
                    steps_stage2=max_steps,
                    Optimizer=optim_cls,
                    Filter=filter_cls,
                    allow_tilt=False,
                    log=f"{out_dir}/relax_{mat_id}.log",
                    enforce_symmetry=enforce_relax_symm,
                )
                atoms.calc = None

    except Exception as exc:
        traceback.print_exc()
        info_dict["errors"].append(f"RelaxError: {exc!r}")
        info_dict["error_traceback"].append(traceback.format_exc())
        kappa_results[mat_id] = info_dict
        continue

    # ---------- Force constants ----------
    try:
        # Ensure required info for phono3py
        atoms.info.setdefault("fc2_supercell", [3, 3, 3])
        # Reduce FC3 supercell if system is large to keep <2000 displacements
        atoms.info.setdefault("fc3_supercell", [2, 2, 2])
        atoms.info.setdefault("q_mesh", [10, 10, 10])

        # Initialize phonon3py
        ph3 = init_phono3py(atoms, log=True, symprec=symprec)

        # Calculate fc2
        ph3, fc2_set, freqs = get_fc2_and_freqs(ph3, calculator=calc, log=True)

        imaginary_freqs = check_imaginary_freqs(freqs)
        freqs_dict = {"imaginary_freqs": imaginary_freqs, "frequencies": freqs}

        # Decide whether to compute LTC
        ltc_condition = not imaginary_freqs and (not relax_dict["broken_symmetry"] or conductivity_broken_symm)

        if ltc_condition:
            # Limit FC3 displacements to 2000 if needed
            ph3, fc3_set = get_fc3(ph3, calculator=calc, log=True)
            if len(fc3_set) > 2000:
                fc3_set = fc3_set[:2000]
        else:
            fc3_set = []

        if save_forces:
            force_results[mat_id] = {"fc2_set": fc2_set, "fc3_set": fc3_set}

        if not ltc_condition:
            info_dict.update(relax_dict)
            info_dict.update(freqs_dict)
            kappa_results[mat_id] = info_dict
            continue

    except Exception as exc:
        traceback.print_exc()
        info_dict["errors"].append(f"ForceConstantError: {exc!r}")
        info_dict["error_traceback"].append(traceback.format_exc())
        info_dict.update(relax_dict)
        kappa_results[mat_id] = info_dict
        continue

    # ---------- Conductivity ----------
    try:
        ph3, kappa_dict = calculate_conductivity(ph3, log=True)
    except Exception as exc:
        traceback.print_exc()
        info_dict["errors"].append(f"ConductivityError: {exc!r}")
        info_dict["error_traceback"].append(traceback.format_exc())
        info_dict.update(relax_dict)
        info_dict.update(freqs_dict)
        kappa_results[mat_id] = info_dict
        continue

    # Merge all results
    kappa_results[mat_id] = {**info_dict, **relax_dict, **freqs_dict, **kappa_dict}

# ---------------- Save results ----------------
df_kappa = pd.DataFrame(kappa_results).T
df_kappa.index.name = ID
df_kappa.reset_index().to_json(out_path)

if save_forces:
    force_out_path = f"{out_dir}/force_sets.json.gz"
    df_force = pd.DataFrame(force_results).T
    df_force = pd.concat([df_kappa, df_force], axis=1)
    df_force.index.name = ID
    df_force.reset_index().to_json(force_out_path)

# ---------------- Save LTC summary CSV ----------------
records = []
for mat_id, data in kappa_results.items():
    kp = data.get("kappa_xx_ave", None)
    kc = data.get("kappa_zz_ave", None)
    ktot = data.get("kappa_TOT_ave", None)
    records.append({
        "mp_id": mat_id,
        "system": data.get("desc", ""),
        "kp": kp,
        "kc": kc,
        "ktot": ktot,
        "imaginary_freqs": data.get("imaginary_freqs", None),
        "errors": data.get("errors", [])
    })

df_summary = pd.DataFrame(records)
summary_csv = os.path.join(out_dir, "thermal_conductivity_summary.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"\n✅ Thermal conductivity summary saved to {summary_csv}")

print(f"\n✅ Done. Results saved in {out_dir}")
