from wannierberri import __utility as util
import pytest
import numpy as np

def test_utility_str2bool():
    for v in "F","f","False","false","fAlSe",".false.":
        assert util.str2bool(v) is False
    for v in "True","true","TRuE",".TrUe.":
        assert util.str2bool(v) is True
    for v in ".true","false.","svas":
        with pytest.raises(ValueError, match=f"unrecognized value of bool parameter :{v}"):
            util.str2bool(v)

def test_utility_FFT():
    inp = np.random.random( (5,4,3,5) ) * (1+2j)
    axes = (0,2)
    for fft in "fftw","numpy", "FfTw", "nUmPY":
        util.FFT(inp, axes,  fft=fft)
    for fft in "fft" , "np",  "idkwhat", "dummy":
        with pytest.raises(ValueError, match=f"unknown type of fft : {fft}"):
            util.FFT(inp, axes,  fft=fft)

def test_utility_FFT_R_to_k():
    NKFFT = (4,5,6)
    iRvec = np.random.randint(100,size=(20,3))
    num_wann = 5
    for lib in 'fftw','numpy', 'NuMpY', 'fftW':
        util.FFT_R_to_k( iRvec, NKFFT, num_wann, lib=lib)

    for lib in 'unknonw','fft','nump','np',"NP":
        with pytest.raises(AssertionError, match=f"fft lib '{lib.lower()}' is unknown/supported"):
            util.FFT_R_to_k( iRvec, NKFFT, num_wann,lib=lib)

    AAA_K = np.random.random( (num_wann,num_wann)+NKFFT+(3,3,3))

    util.FFT_R_to_k( iRvec, NKFFT, num_wann, lib="numpy").transform(AAA_K)
    with pytest.raises(RuntimeError, match="FFT.transform should not be called for slow FT"):
        util.FFT_R_to_k( iRvec, NKFFT, num_wann, lib="slow").transform(AAA_K)

