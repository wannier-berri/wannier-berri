set style data dots
set nokey
set xrange [0: 6.73157]
set yrange [  6.64064 : 43.00464]
set arrow from  1.13571,   6.64064 to  1.13571,  43.00464 nohead
set arrow from  2.52666,   6.64064 to  2.52666,  43.00464 nohead
set arrow from  4.13279,   6.64064 to  4.13279,  43.00464 nohead
set arrow from  4.93586,   6.64064 to  4.93586,  43.00464 nohead
set xtics ("W"  0.00000,"L"  1.13571,"G"  2.52666,"X"  4.13279,"W"  4.93586,"G"  6.73157)
 plot "Pt_band.dat"
