set style data dots
set nokey
set xrange [0: 4.77721]
set yrange [ -4.66722 : 16.87488]
set arrow from  0.90387,  -4.66722 to  0.90387,  16.87488 nohead
set arrow from  1.87188,  -4.66722 to  1.87188,  16.87488 nohead
set arrow from  2.21837,  -4.66722 to  2.21837,  16.87488 nohead
set arrow from  3.25047,  -4.66722 to  3.25047,  16.87488 nohead
set arrow from  4.11348,  -4.66722 to  4.11348,  16.87488 nohead
set xtics ("GAMMA"  0.00000,"L"  0.90387,"U"  1.87188,"X"  2.21837,"GAMMA"  3.25047,"Z"  4.11348,"U"  4.77721)
 plot "GeTe_band.dat"
