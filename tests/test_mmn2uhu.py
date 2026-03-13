import os
from wannierberri.w90files import WannierData
from pytest import approx


def test_mmn2uhu(create_files_Fe_W90):
    data_dir = create_files_Fe_W90
    readnnkp = False
    w90data2 = WannierData.from_w90_files(seedname=os.path.join(data_dir, 'Fe'),
                                            readnnkp=readnnkp,
                                            files=['eig', 'win', 'mmn', 'spn', 'uhu', 'uiu', 'siu', 'shu'])
    w90data1 = WannierData.from_w90_files(seedname=os.path.join(data_dir, 'Fe'),
                                            readnnkp=readnnkp,
                                            files=['eig', 'win', 'mmn', 'spn'])
    w90data1.set_uHu_from_mmn_eig()
    w90data1.set_uIu_from_mmn()
    w90data1.set_sIu_from_mmn_spn()
    w90data1.set_sHu_from_mmn_eig_spn()
    for file in ['uhu', 'uiu', 'siu', 'shu']:
        data1 = w90data1.get_file(file).data
        data2 = w90data2.get_file(file).data
        assert set(data1.keys()) == set(data2.keys()), f"{file} k-point keys mismatch between direct read and sum-over-states construction"
        for ik in data1:
            d1 = data1[ik]
            d2 = data2[ik]
            assert d1 == approx(d2, abs=1e-10), f"{file} data mismatch at k-point {ik}"
