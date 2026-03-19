from ..utility import group_numbers
from .wandata import WannierData


class WannierDataSOC(WannierData):
    """Class to handle Wannier90 data with spin-orbit coupling (SOC)."""

    files_proper = ["soc"]

    def __init__(self, data_up, data_down, soc=None, cell=None):
        self.data_up = data_up
        self.data_down = data_down
        self.bands_were_selected = False
        self._files = {}
        if data_down is None:
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
    def from_npz(cls, seedname, nspin, files=None, irreducible=False):
        """Create Wannier90DataSOC from NPZ files."""
        assert nspin in [1, 2], "nspin must be 1 or 2."
        files_ud = cls.get_files_ud(files)
        data_up = WannierData.from_npz(seedname=seedname + "-spin-0",
                                       files=files_ud,
                                       irreducible=irreducible)
        if nspin == 2:
            data_down = WannierData.from_npz(seedname=seedname + "-spin-1",
                                             files=files_ud,
                                             irreducible=irreducible)
        else:
            data_down = None
        try:
            from .soc import SOC
            soc = SOC.from_npz(seedname + ".soc.npz")
        except FileNotFoundError:
            soc = None
        return cls(data_up=data_up, data_down=data_down, soc=soc)

    def to_npz(self, seedname, files=None):
        """Save Wannier90DataSOC to NPZ files."""
        super().to_npz(seedname=seedname, files=self.get_files_proper(files))
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
                  **kwargs):
        """Create Wannier90DataSOC from a GPAW calculator with SOC."""
        if isinstance(calculator, str):
            from gpaw import GPAW
            calculator = GPAW(calculator, txt=None)
        if spacegroup is None:
            from irrep.spacegroup import SpaceGroup
            spacegroup = SpaceGroup.from_gpaw(calculator)

        cell = {}
        magmoms_on_axis = calculator.get_magnetic_moments()
        cell["magmoms_on_axis"] = group_numbers(magmoms_on_axis, precision=mag_symprec)
        cell["typat"] = calculator.atoms.get_atomic_numbers()
        cell["positions"] = calculator.atoms.get_scaled_positions()

        kwargs_wandata = dict(calculator=calculator,
                              spacegroup=spacegroup,
                              unitary_params=dict(error_threshold=0.1,
                                                  warning_threshold=0.01,
                                                  nbands_upper_skip=8),
                              include_paw=include_paw,
                              include_pseudo=include_pseudo,
                              files=[f for f in files if f not in ["soc", "mmn_ud"]],
                              return_bandstructure=return_bandstructure
                              )
        kwargs_wandata.update(kwargs)
        nspin = calculator.get_number_of_spins()
        if "amn" in files:
            assert projections is not None or (projections_up is not None), \
                "Either projections or projections_up/projections_down must be provided."
            if projections_up is None:
                print("Using 'projections' for both spin up channel.")
                projections_up = projections
            if nspin == 2 and projections_down is None:
                print("No projections_down provided; using projections_up for both spin channels.")
                projections_down = projections_up

        data_up = WannierData.from_gpaw(spin_channel=0,
                                        seedname=seedname + "-spin-0",
                                        projections=projections_up,
                                        **kwargs_wandata)
        if return_bandstructure:
            data_up, bandstructure_up = data_up

        if nspin == 2:
            bkvec = data_up.get_file('bkvec')
            data_down = WannierData.from_gpaw(spin_channel=1,
                                              seedname=seedname + "-spin-1",
                                              projections=projections_down,
                                              bkvec=bkvec,
                                              **kwargs_wandata)
            if return_bandstructure:
                data_down, bandstructure_down = data_down
        else:
            data_down = None
            bandstructure_down = None

        data = cls(data_up=data_up, data_down=data_down, cell=cell)

        if "soc" in files:
            from .soc import SOC
            soc = SOC.from_gpaw(calculator=calculator)
            data.set_file("soc", soc)

        if "mmn_ud" in files and nspin == 2:
            from .mmn import MMN
            bkvec = data_up.get_file('bkvec')
            mmn_ud = MMN.from_bandstructure(bandstructure_left=bandstructure_up,
                                            bandstructure=bandstructure_down,
                                            irreducible=data.is_irreducible,
                                            symmetrizer_left=data_up.get_file("symmetrizer"),
                                            symmetrizer=data_down.get_file("symmetrizer"),
                                            bkvec=bkvec)
            data.set_file("mmn_ud", mmn_ud)

            mmn_du = MMN.from_bandstructure(bandstructure_left=bandstructure_down,
                                            bandstructure=bandstructure_up,
                                            irreducible=data.is_irreducible,
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
            if self.nspin == 2:
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
