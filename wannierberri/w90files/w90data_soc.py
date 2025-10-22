from ..utility import group_numbers
from .soc import SOC
from .w90data import Wannier90data


class Wannier90dataSOC(Wannier90data):
    """Class to handle Wannier90 data with spin-orbit coupling (SOC)."""

    def __init__(self, data_up, data_down, soc=None, cell=None):
        self.data_up = data_up
        self.data_down = data_down
        self.bands_were_selected = False
        self._files = {}
        if data_down is None:
            self.nspin = 1
        else:
            self.nspin = 2
        self.set_file("soc", soc)
        self.cell = cell

    @property
    def is_irreducible(self):
        return self.data_up.irreducible

    @classmethod
    def from_npz(cls, seedname, files=None, irreducible=False):
        """Create Wannier90DataSOC from NPZ files."""

        files_ud = [f for f in files if f != "soc"] if files is not None else None
        data_up = Wannier90data().from_npz(seedname=seedname + "-spin-0",
                                           files=files_ud,
                                           irreducible=irreducible)
        try:
            data_down = Wannier90data().from_npz(seedname=seedname + "-spin-1",
                                                 files=files_ud,
                                                 irreducible=irreducible)
        except FileNotFoundError:
            data_down = None
        try:
            soc = SOC.from_npz(seedname + ".soc.npz")
        except FileNotFoundError:
            soc = None
        return cls(data_up=data_up, data_down=data_down, soc=soc)

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
                  read_npz_list=None,
                  write_npz_list=None,
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

        kwargs_w90data = dict(calculator=calculator,
                              spacegroup=spacegroup,
                              unitary_params=dict(error_threshold=0.1,
                                                  warning_threshold=0.01,
                                                  nbands_upper_skip=8),
                              include_paw=include_paw,
                              include_pseudo=include_pseudo,
                              read_npz_list=read_npz_list,
                              write_npz_list=write_npz_list
                              )
        kwargs_w90data.update(kwargs)
        assert projections is not None or (projections_up is not None), \
            "Either projections or projections_up/projections_down must be provided."
        if projections_up is None:
            print("Using 'projections' for both spin up channel.")
            projections_up = projections
        nspin = calculator.get_number_of_spins()
        if nspin == 2 and projections_down is None:
            print("No projections_down provided; using projections_up for both spin channels.")
            projections_down = projections_up

        data_up = Wannier90data().from_gpaw(spin_channel=0,
                                            seedname=seedname + "-spin-0",
                                            projections=projections_up,
                                            **kwargs_w90data)
        if nspin == 2:
            data_down = Wannier90data().from_gpaw(spin_channel=1,
                                                seedname=seedname + "-spin-1",
                                                projections=projections_down,
                                                **kwargs_w90data)
        else:
            data_down = None
        soc = None
        if read_npz_list is None or "soc" in read_npz_list:
            try:
                soc = SOC.from_npz(seedname + ".soc.npz")
            except FileNotFoundError:
                soc = None
        if soc is None:
            soc = SOC.from_gpaw(calculator=calculator)
        if write_npz_list is None or "soc" in write_npz_list:
            soc.to_npz(seedname + ".soc.npz")
        return cls(data_up=data_up, data_down=data_down, soc=soc, cell=cell)

    def select_bands(self, **kwargs):
        """Select bands for both spin channels."""
        selected_bands_up = self.data_up.select_bands(**kwargs)
        if self.data_down is not None:
            self.data_down.select_bands(selected_bands=selected_bands_up)
        if self.has_file("soc"):
            self.get_file("soc").select_bands(selected_bands_up, selected_bands_up)
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
