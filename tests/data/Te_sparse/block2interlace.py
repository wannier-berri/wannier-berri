import pickle
import numpy as np

parameters = pickle.load(open("parameters_Te_low.pickle", "rb"))

del parameters['symmetrize_info']['DFT_code']
del parameters['use_wcc_phase']
print(parameters.keys())
print(parameters['symmetrize_info'])
print(parameters['wannier_centers_red'])
num_wann = parameters['wannier_centers_red'].shape[0]
nw2 = num_wann // 2
mapping = np.zeros(num_wann, dtype=int)



mapping[:nw2] = np.arange(nw2) * 2
mapping[nw2:] = np.arange(nw2) * 2 + 1
matrices_new = {}
for key, XX in parameters['matrices'].items():
    YY = {}
    print(key)
    for ir, XXR in XX.items():
        # print ("    ", ir)
        YYR = {}
        for (a, b), XXRab in XXR.items():
            # print (f"{a} {b} to {mapping[a]} {mapping[b]} for {ir, key}")
            YYR[mapping[a], mapping[b]] = XXRab
        YY[ir] = YYR
    matrices_new[key] = YY

parameters['matrices'] = matrices_new

wannier_centers_red = parameters['wannier_centers_red']
wannier_centers_red_new = np.zeros_like(wannier_centers_red)
for i in range(num_wann):
    wannier_centers_red_new[mapping[i]] = wannier_centers_red[i]

print(wannier_centers_red_new)
parameters['wannier_centers_red'] = wannier_centers_red_new
pickle.dump(parameters, open("parameters_Te_low_interlaced.pickle", "wb"))
