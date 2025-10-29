import os
import warnings
import numpy as np
import pickle
import torch

from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS

from mattersim.forcefield import MatterSimCalculator

from pymatgen.core import Structure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from pymatgen.io.phonopy import get_phonopy_structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pymatgen.io.ase import AseAtomsAdaptor

from jarvis.core.kpoints import Kpoints3D
from ase import Atoms as AseAtoms
from phonopy.structure.atoms import Atoms as PhonopyAtoms

warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"


def ase_to_phonopy_atoms(ase_atoms, pbc=True):
    return PhonopyAtoms(
        symbols=ase_atoms.symbols,
        positions=ase_atoms.get_positions(),
        pbc=pbc,
        cell=ase_atoms.get_cell(),
    )


def phonopy_to_ase_atoms(phonopy_atoms, pbc=True):
    return AseAtoms(
        symbols=phonopy_atoms.symbols,
        positions=phonopy_atoms.positions,
        pbc=pbc,
        cell=phonopy_atoms.cell,
    )


class mattersim_phonopy:
    """
    Wrapper class to run phonopy workflows using MatterSim forces.
    """

    def __init__(self, structure: Structure, path=".", supercell_dims=(2, 2, 2)):
        self.structure = structure
        self.phonopy_structure = get_phonopy_structure(self.structure)
        self.jarvis_atoms = JarvisAtomsAdaptor.get_atoms(self.structure)
        self.path = path
        self.supercell_dims = list(supercell_dims)
        os.makedirs(self.path, exist_ok=True)
        self.supercell = self.create_supercell()

    def create_supercell(self):
        new_structure = self.structure.copy()
        new_structure.make_supercell(self.supercell_dims)
        supercell_name = os.path.join(
            self.path,
            f"SPOSCAR_{self.supercell_dims[0]}{self.supercell_dims[1]}{self.supercell_dims[2]}",
        )
        new_structure.to(filename=supercell_name)
        return new_structure

    def get_jarvis_kpoints(self, line_density=20):
        return Kpoints3D().kpath(self.jarvis_atoms, line_density=line_density)

    def save_to_pickle(self, filename="mattersim_phonopy_attrs.pkl"):
        filepath = os.path.join(self.path, filename)
        with open(filepath, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def get_phonon_fc2(
        self,
        displacement=0.01,
        num_snapshots=None,
        write_fc=True,
        output_POSCARs=False,
        model_path=None,
        device=device,
    ):
        if model_path and os.path.exists(model_path):
            calc = MatterSimCalculator(model_path=model_path, device=device)
        else:
            calc = MatterSimCalculator(device=device)

        phonon = Phonopy(
            self.phonopy_structure,
            [
                [self.supercell_dims[0], 0, 0],
                [0, self.supercell_dims[1], 0],
                [0, 0, self.supercell_dims[2]],
            ],
        )

        phonon.generate_displacements(distance=displacement, number_of_snapshots=num_snapshots)
        supercells = phonon.supercells_with_displacements

        forces_list = []
        if output_POSCARs:
            disp_dir = os.path.join(self.path, "Displacements")
            os.makedirs(disp_dir, exist_ok=True)

        for idx, scell in enumerate(supercells):
            scell_ase = phonopy_to_ase_atoms(scell, pbc=True)

            if output_POSCARs:
                scell_file = os.path.join(disp_dir, f"POSCAR-{idx+1:03d}")
                scell_ase.write(scell_file, format="vasp", direct=True)

            scell_ase.calc = calc
            forces = scell_ase.get_forces()
            forces -= forces.sum(axis=0) / forces.shape[0]  # drift correction
            forces_list.append(forces)

        phonon.produce_force_constants(forces=forces_list)

        if write_fc:
            fc_dir = os.path.join(self.path, "FORCECONTS")
            os.makedirs(fc_dir, exist_ok=True)
            fc_file = os.path.join(fc_dir, "FORCE_CONSTANTS")
            write_FORCE_CONSTANTS(phonon.force_constants, filename=fc_file)

        self.phonon = phonon

    def get_phonon_dos_bs(
        self,
        line_density=30,
        units="THz",
        output_ph_band=True,
        stability_threshold=-0.1,
        phonopy_bands_dos_figname="phonopy_bands_dos.png",
        dpi=200,
    ):
        """
        Generate phonon band-path frequencies and DOS, save a figure.
        """
        freq_conv = 333.566830 if units == "cm-1" else 1
        kpath = self.get_jarvis_kpoints(line_density=line_density)
        labels = kpath.labels

        freqs, ticks, tick_locs = [], [], []
        prev_k, count, stable = None, 0, True

        for i, k in enumerate(kpath.kpts):
            k_str = ",".join(map(str, k))
            if k_str != prev_k:
                prev_k = k_str
                raw = self.phonon.get_frequencies(k)
                freqs.append(raw)
                if any(f < stability_threshold * freq_conv for f in raw):
                    stable = False
                lbl = labels[i]
                if lbl:
                    ticks.append(f"${lbl}$")
                    tick_locs.append(count)
                count += 1

        # Save stability information to text file only
        stability_status = "stable" if stable else "unstable"
        with open(os.path.join(self.path, "stability"), "w") as sf:
            sf.write(stability_status)

        if output_ph_band:
            freqs = np.array(freqs) * freq_conv

            gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.0)
            plt.rcParams.update({"font.size": 18})
            plt.figure(figsize=(10, 5))

            ax = plt.subplot(gs[0])
            # Plot all bands in blue color
            for b in range(freqs.shape[1]):
                ax.plot(freqs[:, b], lw=2, color='blue', alpha=0.7)
            for x in tick_locs:
                ax.axvline(x=x, c="black")
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(ticks)
            ax.set_ylabel(f"Frequency ({units})")
            ax.set_xlim([0, max(tick_locs)])

            # run DOS on mesh - FIXED: use correct phonopy API
            self.phonon.run_mesh([10, 10, 10], is_gamma_center=True, is_mesh_symmetry=False)
            self.phonon.run_total_dos()
            dos = self.phonon.get_total_dos_dict()

            # Extract frequencies and DOS values correctly
            dos_freqs = np.array(dos['frequency_points']) * freq_conv
            dos_vals = np.array(dos['total_dos'])

            # Set reasonable frequency limits
            min_freq = min(np.min(freqs), np.min(dos_freqs))
            max_freq = max(np.max(freqs), np.max(dos_freqs))

            # Ensure we don't have negative infinity issues
            min_freq = max(min_freq, -10)  # Set a reasonable lower bound

            ax.set_ylim([min_freq, max_freq])

            ax2 = plt.subplot(gs[1])
            ax2.fill_between(dos_vals, dos_freqs, color=(0.2, 0.4, 0.6, 0.6),
                           edgecolor="k", alpha=0.6)
            ax2.set_xlabel("DOS")
            ax2.set_yticks([])
            ax2.set_xticks([])
            ax2.set_ylim([min_freq, max_freq])
            ax2.set_xlim([0, max(dos_vals) * 1.1])

            outfn = os.path.join(self.path, phonopy_bands_dos_figname)
            os.makedirs(os.path.dirname(outfn) if os.path.dirname(outfn) else '.', exist_ok=True)
            plt.tight_layout()
            plt.savefig(outfn, dpi=dpi)
            plt.close()

        return stability_status

    def generate_bands_conf(self, filename="orig_band.conf", line_density=30, BAND_POINTS=100):
        kpath = Kpoints3D().kpath(self.jarvis_atoms, line_density=line_density)
        coords, labels = kpath._kpoints, kpath._labels

        lines = " ".join(f"{x} {y} {z}" for x, y, z in coords)
        atom_names = " ".join(sorted(set(self.jarvis_atoms.elements)))

        with open(os.path.join(self.path, filename), "w") as f:
            f.write("PRIMITIVE_AXES = AUTO\n")
            f.write(f"ATOM_NAME = {atom_names}\n")
            f.write("DIM = " + " ".join(map(str, self.supercell_dims)) + "\n")
            f.write("FORCE_CONSTANTS = READ\n")
            f.write("BAND= " + lines + "\n")

        ase_atoms = AseAtomsAdaptor.get_atoms(self.structure)
        bp = ase_atoms.cell.bandpath()._kpts
        lines2 = " ".join(f"{x} {y} {z}" for x, y, z in bp)

        with open(os.path.join(self.path, "band.conf"), "w") as f:
            f.write("PRIMITIVE_AXES = AUTO\n")
            f.write(f"ATOM_NAME = {atom_names}\n")
            f.write("DIM = " + " ".join(map(str, self.supercell_dims)) + "\n")
            f.write("FORCE_CONSTANTS = READ\n")
            f.write("BAND= " + lines2 + "\n")
            f.write(f"BAND_POINTS = {BAND_POINTS}\n")


if __name__ == "__main__":
    pmg_struc = Structure.from_file("POSCAR")
    mp = mattersim_phonopy(pmg_struc, path=".")
    mp.save_to_pickle()
    mp.get_phonon_fc2(output_POSCARs=True, write_fc=True, model_path="mattersim-v1.0.0-1M.pth", device=device)
    mp.get_phonon_dos_bs()
