set multiplot layout 1,2

set xlabel "frequency (eV)"
set ylabel "SHC (S/cm)"

plot 'pt-opt_SHCqiao_iter-0000_freqscan.dat' u 2:14 w l title "SHCqiao" ,\
     'pt-opt_SHCryoo_iter-0000_freqscan.dat' u 2:14 w l title "SHCryoo"

set xlabel "Fermi energy shift (eV)"
unset ylabel

plot 'pt-opt_SHCqiao_iter-0000_fermiscan.dat' u (($1)-18.1299):14 w l title "SHCqiao" ,\
     'pt-opt_SHCryoo_iter-0000_fermiscan.dat' u (($1)-18.1299):14 w l title "SHCryoo"