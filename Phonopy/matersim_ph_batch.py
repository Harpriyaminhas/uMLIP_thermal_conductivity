import os
import glob
import traceback
import pandas as pd
from pymatgen.core import Structure
from mattersim_phonopy_class import mattersim_phonopy

# Path containing all POSCARs
poscar_dir = "./POSCARs_mp"
poscar_files = sorted(glob.glob(os.path.join(poscar_dir, "POSCAR-mp-*")))

# Output directory for results
results_dir = "./results_mattersim"
os.makedirs(results_dir, exist_ok=True)

# MatterSim model path
model_path = "mattersim-v1.0.0-1M.pth"

# Counter for tracking progress
success_count = 0
error_count = 0
skipped_oxygen_count = 0

# List to collect stability information for summary CSV
stability_data = []

print(f"Found {len(poscar_files)} POSCAR files to process")

for i, poscar_file in enumerate(poscar_files):
    try:
        print(f"\n[{i+1}/{len(poscar_files)}] Processing {poscar_file} ...")

        # Load structure and check for Oxygen
        structure = Structure.from_file(poscar_file)
        elements = [element.symbol for element in structure.composition.elements]
        
        # Skip systems containing Oxygen
        if 'O' in elements:
            skipped_oxygen_count += 1
            mp_id = os.path.basename(poscar_file).replace("POSCAR-", "").replace("mp-", "")
            formula = structure.composition.reduced_formula
            print(f"⚠ Skipping Oxygen-containing system: {mp_id} ({formula})")
            
            stability_data.append({
                'mp_id': mp_id,
                'formula': formula,
                'stability': 'skipped_oxygen'
            })
            continue

        mp_id = os.path.basename(poscar_file).replace("POSCAR-", "").replace("mp-", "")
        formula = structure.composition.reduced_formula

        # Create output folder per material
        out_path = os.path.join(results_dir, f"mp-{mp_id}")
        os.makedirs(out_path, exist_ok=True)

        # Save original POSCAR in output directory
        structure.to(filename=os.path.join(out_path, "POSCAR"))

        # Initialize phonopy wrapper
        ms_ph = mattersim_phonopy(structure, path=out_path, supercell_dims=(3, 3, 3))

        # Save object for reference
        ms_ph.save_to_pickle(filename=f"mattersim_phonopy_{mp_id}.pkl")

        # Generate displaced supercells, compute forces, and write FORCE_CONSTANTS
        ms_ph.get_phonon_fc2(
            displacement=0.01,
            output_POSCARs=True,
            write_fc=True,
            model_path=model_path,
        )

        # Generate phonon band structure and DOS figure + save data files
        stability_status = ms_ph.get_phonon_dos_bs(
            line_density=30,
            units="THz",
            output_ph_band=True,
            stability_threshold=-0.1,
            phonopy_bands_dos_figname=f"phonopy_bands_dos_{mp_id}.png",
            dpi=200
        )

        # Collect stability data for summary
        stability_data.append({
            'mp_id': mp_id,
            'formula': formula,
            'stability': stability_status
        })

        success_count += 1
        print(f"✓ Successfully completed: {mp_id}")

    except KeyboardInterrupt:
        # Handle manual interruption (Ctrl+C)
        print(f"\n⚠ Processing interrupted by user. Moving to next system...")
        error_count += 1
        
        try:
            mp_id = os.path.basename(poscar_file).replace("POSCAR-", "").replace("mp-", "")
            formula = structure.composition.reduced_formula if 'structure' in locals() else "unknown"
        except:
            mp_id = "unknown"
            formula = "unknown"

        stability_data.append({
            'mp_id': mp_id,
            'formula': formula,
            'stability': 'interrupted'
        })
        
        # Continue to next system instead of breaking
        continue
        
    except Exception as e:
        error_count += 1
        print(f"✗ Error processing {poscar_file}: {e}")
        print(traceback.format_exc())

        # Also add failed systems to the CSV with error status
        try:
            mp_id = os.path.basename(poscar_file).replace("POSCAR-", "").replace("mp-", "")
            structure = Structure.from_file(poscar_file)
            formula = structure.composition.reduced_formula
        except:
            mp_id = "unknown"
            formula = "unknown"

        stability_data.append({
            'mp_id': mp_id,
            'formula': formula,
            'stability': 'error'
        })

        # Save error information
        try:
            error_file = os.path.join(results_dir, f"error_{mp_id}.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error processing {poscar_file}\n")
                f.write(f"Error: {e}\n")
                f.write(traceback.format_exc())
        except:
            pass

        # Continue to next system
        continue

# Save single comprehensive CSV with all systems at the end
if stability_data:
    summary_df = pd.DataFrame(stability_data)
    summary_csv_path = os.path.join(results_dir, "all_systems_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nComprehensive CSV saved to: {summary_csv_path}")
    print(f"Total systems in CSV: {len(summary_df)}")
    
    # Count different status types
    stable_count = len(summary_df[summary_df['stability'] == 'stable'])
    unstable_count = len(summary_df[summary_df['stability'] == 'unstable'])
    error_count_csv = len(summary_df[summary_df['stability'] == 'error'])
    interrupted_count = len(summary_df[summary_df['stability'] == 'interrupted'])
    skipped_count = len(summary_df[summary_df['stability'] == 'skipped_oxygen'])
    
    print(f"Breakdown:")
    print(f"  Stable: {stable_count}")
    print(f"  Unstable: {unstable_count}")
    print(f"  Errors: {error_count_csv}")
    print(f"  Interrupted: {interrupted_count}")
    print(f"  Skipped (Oxygen): {skipped_count}")

print(f"\n=== Summary ===")
print(f"Successfully processed: {success_count}")
print(f"Errors: {error_count}")
print(f"Skipped (Oxygen systems): {skipped_oxygen_count}")
print(f"Total POSCARs found: {len(poscar_files)}")
