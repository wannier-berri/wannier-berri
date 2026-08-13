import os

import numpy as np

from ..utility import group_numbers
from .wandata import WannierData


class WannierDataSOC(WannierData):
    """Class to handle Wannier90 data with spin-orbit coupling (SOC)."""

    files_proper = ["soc", "cell", "mmn_ud", "mmn_du"]

    def __init__(self, data_up, data_down, soc=None, cell=None, seedname="wannier_soc", altermagnetic=False):
        self.data_up = data_up
        self.data_down = data_down
        self.bands_were_selected = False
        self._files = {}
        self.seedname = seedname
        self.altermagnetic = altermagnetic
        if data_down is None and not altermagnetic:
            self.nspin = 1
        else:
            self.nspin = 2
        if soc is not None:
            self.set_file("soc", soc)
        self.cell = cell

    @property
    def irreducible(self):
        return self.data_up.irreducible

    @classmethod
    def get_files_ud(cls, files):
        if files is None:
            return None
        else:
            return [f for f in files if f not in cls.files_proper]

    @classmethod
    def get_files_proper(cls, files):
        if files is None:
            return None
        else:
            return [f for f in files if f in cls.files_proper]

    @classmethod
    def from_npz(cls, seedname, nspin=None, files=None, irreducible=False, ignore_missing_files=False):
        """Create Wannier90DataSOC from NPZ files."""
        cell = np.load(seedname + ".cell.npz", allow_pickle=True)
        if nspin is None:
            if 'nspin' in cell:
                nspin = cell["nspin"]
            else:
                raise ValueError("nspin must be provided if not stored in .cell.npz file.")
        altermagnetic = "altermagnetic_rotation_latt" in cell
        assert nspin in [1, 2], f"nspin must be 1 or 2. Got {nspin}."

        files_ud = cls.get_files_ud(files)
        data_up = WannierData.from_npz(seedname=seedname + "-spin-0",
                                       files=files_ud,
                                       irreducible=irreducible,
                                       ignore_missing_files=ignore_missing_files)
        if nspin == 2 and not altermagnetic:
            data_down = WannierData.from_npz(seedname=seedname + "-spin-1",
                                             files=files_ud,
                                             irreducible=irreducible,
                                             ignore_missing_files=ignore_missing_files)
        else:
            data_down = None
        try:
            from .soc import SOC
            soc = SOC.from_npz(seedname + ".soc.npz")
        except FileNotFoundError:
            if ignore_missing_files:
                print(f"Warning: SOC file {seedname}.soc.npz not found. SOC will be set to None.")
                soc = None
            else:
                raise FileNotFoundError(f"SOC file {seedname}.soc.npz not found.")

        data_soc = cls(data_up=data_up, data_down=data_down, soc=soc, seedname=seedname, altermagnetic=altermagnetic)
        if os.path.isfile(seedname + ".cell.npz"):
            cell = np.load(seedname + ".cell.npz", allow_pickle=True)
            data_soc.cell = {key: val for key, val in cell.items()}
        return data_soc

    def to_npz(self, seedname=None, files=None):
        if seedname is None:
            seedname = self.seedname
        """Save Wannier90DataSOC to NPZ files."""
        super().to_npz(seedname=seedname, files=self.get_files_proper(files))
        if self.cell is not None:
            np.savez(seedname + ".cell.npz", **self.cell)
        files_ud = [f for f in files if f != "soc"] if files is not None else None
        self.data_up.to_npz(seedname=seedname + "-spin-0", files=files_ud)
        if self.data_down is not None:
            self.data_down.to_npz(seedname=seedname + "-spin-1", files=files_ud)


    @classmethod
    def from_gpaw(cls, calculator,
                  projections=None,
                  projections_up=None,
                  projections_down=None,
                  seedname="wannier_soc",
                  spacegroup=None,
                  mag_symprec=0.05,
                  include_paw=True,
                  include_pseudo=True,
                  files=["mmn", "eig", "amn", "symmetrizer", "soc"],
                  return_bandstructure=False,
                  altermagnetic=False,
                  altermagnetic_nskip_symmetries=0,
                  **kwargs):
        """Create Wannier90DataSOC from a GPAW calculator with SOC.

        """
        if isinstance(calculator, str):
            from gpaw import GPAW
            calculator = GPAW(calculator, txt=None)
        if spacegroup is None:
            from irrep.spacegroup import SpaceGroup
            spacegroup = SpaceGroup.from_gpaw(calculator)

        nspin = calculator.get_number_of_spins()
        cell = {}
        magmoms_on_axis = calculator.get_magnetic_moments()
        cell["magmoms_on_axis"] = group_numbers(magmoms_on_axis, precision=mag_symprec)
        cell["typat"] = calculator.atoms.get_atomic_numbers()
        cell["positions"] = calculator.atoms.get_scaled_positions()
        cell["nspin"] = nspin
        if altermagnetic:
            if "symmetrizer" not in files:
                files.append("symmetrizer")

        kwargs_wandata = dict(calculator=calculator,
                              spacegroup=spacegroup,
                              unitary_params=dict(error_threshold=0.1,
                                                  warning_threshold=0.01,
                                                  nbands_upper_skip=8),
                              include_paw=include_paw,
                              include_pseudo=include_pseudo,
                              files=[f for f in files if f not in ["soc", "mmn_ud"]],
                              )
        return_bandstructure_loc = return_bandstructure or ("mmn_ud" in files and nspin == 2) or "soc" in files
        return_paw = include_paw or ("mmn_ud" in files and nspin == 2) or "soc" in files
        kwargs_wandata.update(kwargs)
        if "amn" in files:
            assert projections is not None or (projections_up is not None), \
                "Either projections or projections_up/projections_down must be provided."
            if projections_up is None:
                print("Using 'projections' for both spin up channel.")
                projections_up = projections
            if nspin == 2 and projections_down is None and not altermagnetic:
                print("No projections_down provided; using projections_up for both spin channels.")
                projections_down = projections_up

        data_up = WannierData.from_gpaw(spin_channel=0,
                                        seedname=seedname + "-spin-0",
                                        projections=projections_up,
                                        return_bandstructure=return_bandstructure_loc,
                                        return_paw=return_paw,
                                        **kwargs_wandata)
        if return_bandstructure_loc:
            data_up, bandstructure_up = data_up
        if altermagnetic:
            from irrep.altermagnetic_transformer import AltermagneticTransformer
            symmetrizer_up = data_up.get_file("symmetrizer")
            altermagnetic_transformer = AltermagneticTransformer.from_gpaw(calculator,
                                                                           symmetrizer_up=symmetrizer_up,
                                                                           nskip_symmetries=altermagnetic_nskip_symmetries)
            alter_symop = altermagnetic_transformer.alter_symop
            cell["altermagnetic_rotation_latt"] = alter_symop.rotation
            cell["altermagnetic_translation_latt"] = alter_symop.translation
            cell["altermagnetic_k_mapping"] = altermagnetic_transformer.alter_map
            cell["altermagnetic_k_mapping_isym"] = altermagnetic_transformer.alter_map_isym
            nspin = 1
        else:
            altermagnetic_transformer = None



        if nspin == 2 and not altermagnetic:
            bkvec = data_up.get_file('bkvec')
            data_down = WannierData.from_gpaw(spin_channel=1,
                                              seedname=seedname + "-spin-1",
                                              projections=projections_down,
                                              bkvec=bkvec,
                                              return_bandstructure=return_bandstructure_loc,
                                              return_paw=return_paw,
                                              **kwargs_wandata)
            if return_bandstructure_loc:
                data_down, bandstructure_down = data_down
        else:
            data_down = None
            bandstructure_down = None

        data = cls(data_up=data_up, data_down=data_down, cell=cell, seedname=seedname)

        if "soc" in files:
            from .soc import SOC
            # check if irrep version is smalle 3.2
            # from packaging import version
            # import irrep
            if False:  # version.parse(irrep.__version__) < version.parse("3.1.2") and not altermagnetic:
                soc = SOC.from_gpaw(calculator=calculator,
                                    IBstart=kwargs.get("IBstart", None),
                                    IBend=kwargs.get("IBend", None),)
            else:
                soc = SOC.from_bandstructure(bandstructure_up=bandstructure_up,
                                            bandstructure_down=bandstructure_down,
                                            altermagnetic_transformer=altermagnetic_transformer)
            data.set_file("soc", soc)

        if "mmn_ud" in files and nspin == 2:
            if altermagnetic:
                raise NotImplementedError("Altermagnetic symmetry is not yet implemented for mmn_ud generation.")
            from .mmn import MMN
            bkvec = data_up.get_file('bkvec')
            mmn_ud = MMN.from_bandstructure(bandstructure_left=bandstructure_up,
                                            bandstructure=bandstructure_down,
                                            irreducible=data.irreducible,
                                            symmetrizer_left=data_up.get_file("symmetrizer"),
                                            symmetrizer=data_down.get_file("symmetrizer"),
                                            bkvec=bkvec)
            data.set_file("mmn_ud", mmn_ud)

            mmn_du = MMN.from_bandstructure(bandstructure_left=bandstructure_down,
                                            bandstructure=bandstructure_up,
                                            irreducible=data.irreducible,
                                            symmetrizer_left=data_down.get_file("symmetrizer"),
                                            symmetrizer=data_up.get_file("symmetrizer"),
                                            bkvec=bkvec,)
            data.set_file("mmn_du", mmn_du)

        if return_bandstructure:
            return data, (bandstructure_up, bandstructure_down) if nspin == 2 else bandstructure_up
        else:
            return data


    def select_bands(self, **kwargs):
        """Select bands for both spin channels."""
        selected_bands_up = self.data_up.select_bands(**kwargs)
        if self.data_down is not None:
            self.data_down.select_bands(selected_bands=selected_bands_up)
        if self.has_file("soc"):
            self.get_file("soc").select_bands(selected_bands_up, selected_bands_up)
        if self.has_file("mmn_ud"):
            self.get_file("mmn_ud").select_bands(selected_bands=selected_bands_up)
        if self.has_file("mmn_du"):
            self.get_file("mmn_du").select_bands(selected_bands=selected_bands_up)
        self.bands_were_selected = True

    def wannierise(self, ispin=None, **kwargs):
        if ispin == 0:
            self.data_up.wannierise(**kwargs)
        elif ispin == 1:
            if self.data_down is None:
                raise ValueError("No spin-down data available for wannierisation.")
            self.data_down.wannierise(**kwargs)
        elif ispin is None:
            self.wannierise(ispin=0, **kwargs)
            if self.data_down is not None:
                self.wannierise(ispin=1, **kwargs)
        else:
            raise ValueError(f"Invalid ispin value: {ispin}. Must be 0, 1, or None.")

    def get_file_ud(self, updown, key):
        assert updown in ["up", "down"]
        if updown == 'up' or self.data_down is None:
            return self.data_up.get_file(key)
        else:
            return self.data_down.get_file(key)

    def set_projections(self,
                        projections=None,
                        projections_up=None,
                        projections_down=None,
                        bandstructure=None,
                        bandstructure_up=None,
                        bandstructure_down=None,
                        **kwargs
                        ):
        if projections is not None:
            print("Using 'projections' for both spin channels.")
            if projections_up is not None:
                print("Warning: 'projections' will override 'projections_up'.")
            projections_up = projections
            if projections_down is not None:
                print("Warning: 'projections' will override 'projections_down'.")
            projections_down = projections
        if self.nspin == 2:
            assert bandstructure_up is not None and bandstructure_down is not None, "two bandstructures (up and down) must be provided for nspin=2."
        elif self.nspin == 1:
            if bandstructure is not None:
                if bandstructure_up is not None:
                    Warning("bandstructure_up will be ignored since nspin=1., using `bandstructure` instead.")
                bandstructure_up = bandstructure
        self.data_up.set_projections(projections=projections_up,
                                     bandstructure=bandstructure_up,
                                     **kwargs)
        if self.nspin == 2:
            self.data_down.set_projections(projections=projections_down,
                                           bandstructure=bandstructure_down,
                                           **kwargs)
