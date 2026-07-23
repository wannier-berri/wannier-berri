"""Test the select_window_degen function"""

import numpy as np
import pytest

from wannierberri.utility import select_window_degen

# bands 6 and 7 are degenerate (gap 1 meV << thresh = 10 meV)
E_DEGEN = np.array([-5., -4., -3., -2., -1., 0., 1., 1.001, 3., 4.])
# bands 1 and 2 are degenerate -- lower edge
E_DEGEN_LOW = np.array([-5., -4.001, -4.0, -2., -1., 0., 1., 2., 3., 4.])
 
@pytest.mark.parametrize("include_degen", [True, False])
def test_edge_cutting_multiplet_does_not_raise(include_degen):
    """Window edge inside a degenerate pair must not raise."""
    select_window_degen(E_DEGEN, win_min=-6, win_max=1.0,
                        include_degen=include_degen, return_indices=True)

@pytest.mark.parametrize("include_degen", [True, False])
@pytest.mark.parametrize("win_min", [-6.0, -2.5])   # selection from band 0
def test_indices_are_unique_and_sorted(include_degen, win_min):
    """The returned index list must be a valid, duplicate-free band selection."""
    ind = select_window_degen(E_DEGEN, win_min=win_min, win_max=1.0,
                              include_degen=include_degen, return_indices=True)
    ind = list(ind)
    assert len(ind) == len(set(ind)), f"duplicated band indices: {ind}"
    assert ind == sorted(ind)
    assert all(0 <= i < len(E_DEGEN) for i in ind)

@pytest.mark.parametrize("include_degen", [True, False])
@pytest.mark.parametrize("win_max", [0.5, 1.0, 2.0, 5.0])
def test_mask_and_indices_agree(include_degen, win_max):
    """return_indices=True and the boolean mask must describe the same set."""
    ind = select_window_degen(E_DEGEN, win_min=-6, win_max=win_max,
                              include_degen=include_degen, return_indices=True)
    mask = select_window_degen(E_DEGEN, win_min=-6, win_max=win_max,
                               include_degen=include_degen, return_indices=False)
    assert np.array_equal(np.where(mask)[0], np.array(sorted(ind), dtype=int))
 
 
@pytest.mark.parametrize("include_degen", [True, False])
@pytest.mark.parametrize("E,pair,win", [
    # upper window edge cutting the degenerate pair (bands 6,7)
    *[(E_DEGEN, (6, 7), (-6.0, wmax)) for wmax in (0.999, 1.0, 1.0005, 1.001)],
    # lower window edge cutting the degenerate pair (bands 1,2)
    *[(E_DEGEN_LOW, (1, 2), (wmin, 5.0)) for wmin in (-4.002, -4.0005, -4.0)],
])
def test_degenerate_pair_not_split(E, pair, win, include_degen):
    """A degenerate multiplet is either fully in or fully out of the selection,
    whether it straddles the upper or the lower window edge."""
    i, j = pair
    win_min, win_max = win
    mask = select_window_degen(E, win_min=win_min, win_max=win_max,
                               include_degen=include_degen)
    assert mask[i] == mask[j], (
        f"bands {i},{j} are degenerate but the selection split them "
        f"(window=({win_min}, {win_max}), include_degen={include_degen})")
 
 
def test_unaffected_cases_unchanged():
    """Edges sitting in real gaps keep the original behaviour."""
    ind = select_window_degen(E_DEGEN, win_min=-6, win_max=0.5,
                              include_degen=True, return_indices=True)
    assert list(ind) == [0, 1, 2, 3, 4, 5]
    ind = select_window_degen(E_DEGEN, win_min=-6, win_max=5.0,
                              include_degen=True, return_indices=True)
    assert list(ind) == list(range(len(E_DEGEN)))
 
 
def test_empty_window():
    assert select_window_degen(E_DEGEN, win_min=10, win_max=20,
                               return_indices=True) == []
    assert not select_window_degen(E_DEGEN, win_min=10, win_max=20).any()

def test_degeneracy_between_bands_0_and_1():
    E = np.array([-4.001, -4.0, -2., -1., 0., 1.])   # bands 0,1 degenerate
    mask = select_window_degen(E, win_min=-4.0005, win_max=2.0, include_degen=True)
    assert mask[0] == mask[1], "degeneracy between bands 0 and 1 was not handled"