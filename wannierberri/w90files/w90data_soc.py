import numpy as np

from ..utility import group_numbers
from .soc import SOC
from .w90data import Wannier90data


class Wannier90DataSOC(Wannier90data):
    """Class to handle Wannier90 data with spin-orbit coupling (SOC)."""

    def __init__(self, data_up, data_down, soc=None, cell=None):
        self.data_up = data_up
        self.data_down = data_down
        if data_down is None:
            self.nspin = 1
        else:
            self.nspin = 2
        self.soc = soc
        self.cell = cell

    @property
    def is_irreducible(self):
        return self.data_up.irreducible

    @classmethod
    def from_gpaw(cls, calc,
                  projections=None,
                  projections_up=None,
                  projections_down=None,
                  seedname="wannier_soc",
                  spacegroup=None,
                  mag_symprec=0.05,
                  **kwargs):
        """Create Wannier90DataSOC from a GPAW calculator with SOC."""
        if isinstance(calc, str):
            from gpaw import GPAW
            calc = GPAW(calc, txt=None)
        if spacegroup is None:
            from irrep.spacegroup import SpaceGroup
            spacegroup = SpaceGroup.from_gpaw(calc)

        cell = {}
        magmoms_on_axis = calc.get_magnetic_moments()
        cell["magmoms_on_axis"] = group_numbers(magmoms_on_axis, precision=mag_symprec)
        cell["typat"] = calc.atoms.get_atomic_numbers()
        cell["positions"] = calc.atoms.get_scaled_positions()

        kwargs_w90data = dict(calculator=calc,
                              mp_grid=(2, 2, 2),
                              read_npz_list=[],
                              spacegroup=spacegroup,
                              unitary_params=dict(error_threshold=0.1,
                                                  warning_threshold=0.01,
                                                  nbands_upper_skip=8),
                              )
        kwargs_w90data.update(kwargs)
        assert projections is not None or (projections_up is not None), \
            "Either projections or projections_up/projections_down must be provided."
        if projections_up is None:
            projections_up = projections
        nspin = calc.get_number_of_spins()
        if nspin == 2 and projections_down is None:
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
        soc = SOC.from_gpaw(calc)
        return cls(data_up=data_up, data_down=data_down, soc=soc, cell=cell)

    def select_bands(self, **kwargs):
        """Select bands for both spin channels."""
        selected_bands = self.data_up.select_bands(**kwargs)
        if self.data_down is not None:
            selected_bands_2 = self.data_down.select_bands(selected_bands=selected_bands)
            assert np.all(selected_bands == selected_bands_2), \
                "Selected bands for spin-up and spin-down channels do not match."
        if self.soc is not None:
            self.soc.select_bands(selected_bands)

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
