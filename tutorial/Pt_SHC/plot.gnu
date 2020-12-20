set xlabel "frequency (eV)"
set ylabel "SHC (S/cm)"

plot 'pt-opt_SHCqiao-freqscan_iter-0000.dat' u 1:13 w l title "SHCqiao"
replot 'pt-opt_SHCryoo-freqscan_iter-0000.dat' u 1:13 w l title "SHCryoo"