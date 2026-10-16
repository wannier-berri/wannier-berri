from wannierberri.system import System_R
projname = "s,p"
system_bulk = System_R.from_npz(f"diamond-bulk-{projname}")
nslab = 10



system_slab = system_bulk.make_slab([[-1, 1, 0], [0, 0, 1], [1, 1, -1]], nslab=nslab)
system_slab.exclude_WF_mask("*_cell-0")
print("wannier names in slab system: ", system_slab.wannier_names)
print("real lattice vectors in slab system: ", system_slab.real_lattice)
print("wannier centers in slab system: ", system_slab.wannier_centers_red)

path_surface, bands_surface = system_slab.get_bandstructure(dk=0.05)
bands_surface.plot_path_fat(path_surface,
                        close_fig=False,
                        show_fig=False,
                        linecolor='k',
                        label=projname,
                        Emin=-10,
                        Emax=50,
                        save_file=f'diamond-slab-{projname}.pdf',
                        )
