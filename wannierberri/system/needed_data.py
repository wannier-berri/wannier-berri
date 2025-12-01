from collections import defaultdict


class NeededData:

    needed_files = defaultdict(lambda: [])
    needed_files['AA'] = ['mmn']
    needed_files['BB'] = ['mmn', 'eig']
    needed_files['CC'] = ['uhu', 'mmn']
    needed_files['OO'] = ['uiu', 'mmn']  # mmn is needed here because it stores information on
    needed_files['GG'] = ['uiu', 'mmn']  # neighboring k-points
    needed_files['SS'] = ['spn']
    needed_files['SH'] = ['spn', 'eig']
    needed_files['SR'] = ['spn', 'mmn']
    needed_files['SA'] = ['siu', 'mmn']
    needed_files['SHA'] = ['shu', 'mmn']

    @classmethod
    def get_parameters(cls, **parameters):
        """selects the parameters needed for this class from a dictionary of parameters,
        and removes them from the dictionary"""

        remove_keys = []
        return_dict = {}
        for key, val in parameters.items():
            if key in ["berry", "morb", "spin",
                       "SHCryoo", "SHCqiao",
                       "OSD", "_getFF",
                       "force_internal_terms_only",
                       "chk"]:
                return_dict[key] = val
                if key not in ["force_internal_terms_only"]:
                    remove_keys.append(key)
        for key in remove_keys:
            del parameters[key]
        return parameters, return_dict

    def __init__(self,
                 berry=False,
                 morb=False,
                 spin=False,
                 SHCryoo=False, SHCqiao=False,
                 OSD=False,
                 _getFF=False,
                 force_internal_terms_only=False,
                 chk=True,
                 **kwargs):
        self.matrices = {'Ham'}
        self.files = set()
        if morb:
            self.matrices.update(['AA', 'BB', 'CC'])
        if berry:
            self.matrices.add('AA')
        if spin:
            self.matrices.add('SS')
        if _getFF:
            self.matrices.add('FF')
        if SHCryoo:
            self.matrices.update(['AA', 'SS', 'SA', 'SHA', 'SH'])
        if SHCqiao:
            self.matrices.update(['AA', 'SS', 'SR', 'SH', 'SHR'])
        if OSD:
            self.matrices.update(['AA', 'BB', 'CC', 'GG', 'OO'])
        if force_internal_terms_only:
            self.matrices = self.matrices.intersection(['Ham', 'SS'])
        for mat in self.matrices:
            self.files.update(self.needed_files[mat])
        if chk:
            self.files.add('chk')


    def need_any(self, keys):
        """returns True is any of the listed matrices is needed in to be set

        keys : str or list of str
            'AA', 'BB', 'CC', etc
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for k in keys:
            if k in self.matrices:
                return True


    def not_in_list(self, matrices):
        """returns the set of matrices which are needed but not in the given list or set"""
        return set(self.matrices) - set(matrices)
