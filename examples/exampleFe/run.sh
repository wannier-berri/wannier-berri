wannier90_path="../../../wannier90"
wannier90=$wannier90_path"/wannier90.x"
postw90=$wannier90_path"/postw90.x"


tar -zvf Fe_wan_files.tar.gz

EFERMI=12.6

NK_FFT=15
NK_div=2
NK_tot=$((NK_FFT*NK_div))

echo "wanierizing"


cp Fe.win0 Fe.win
$wannier90 Fe

rm Fe_wsvec.dat Fe_band* 

echo
echo "evaluating AHC using wannier19 from Fe_tb.dat"
time ./calc_AHC.py tb $NK_FFT $NK_div $EFERMI

echo
echo "evaluating AHC using postw90"

echo  "berry = true"> Fe.win 
echo "fermi_energy = $EFERMI" >> Fe.win 
echo "berry_task = ahc">> Fe.win 
echo "berry_kmesh =  $NK_tot $NK_tot $NK_tot" >> Fe.win
cat Fe.win0 >> Fe.win

time $postw90 Fe

echo
echo "The postw90 results:"

tail -30 Fe.wpout


#### to run the following partone needs to compile postw90 from the following repository:
#### https://github.com/stepan-tsirkin/wannier90/tree/saveHH

echo
echo "saving AA_R, HH_R"

echo  "get_oper_save=T"> Fe.win 
echo "get_oper_save_task=a" >> Fe.win 
cat Fe.win0 >> Fe.win
time $postw90 Fe

echo
echo "evaluating AHC using wannier19 from Fe_AA_R.dat, Fe_HH_R.dat, Fe_HH_save.info"
time ./calc_AHC.py aa $NK_FFT $NK_div $EFERMI
