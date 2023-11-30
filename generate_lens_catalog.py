import numpy as np
from astropy.table import Table
from tqdm import tqdm
from rotate import *


def calc_ellipticity(data):
    """
    Calculates the ellipticity of a galaxy from the data.

    data: astropy table
        Table containing the data of the galaxy
        necessary columns: 
            ra, dec, eigenvalues, eigenvectors1, eigenvectors2, eigenvectors3 
    """

    table_l = Table()
    table_l['Mh']   = data['MassHMsun']
    table_l['ra']   = data['RA']
    table_l['dec']  = data['DEC']
    table_l['z']    = data['Z_COSMO']

    n = len(data)
    table_l['w']   = np.ones(n)
    table_l['e_1'] = np.ones(n)
    table_l['e_2'] = np.ones(n)

    for i in tqdm(range(n)):
        main_tensor = get_inertia_tensor(data['sigman_L2com'][i])
        coordbase1 = np.array([
            data['sigman_eigenvecsMaj_L2com'][i], 
            data['sigman_eigenvecsMid_L2com'][i], 
            data['sigman_eigenvecsMin_L2com'][i]
        ]).T
        coordbase2 = get_coordbase_parallel_ra_dec(table_l['ra'][i], table_l['dec'][i])

        inertia_tensor = inertia_tensor_transform(main_tensor, coordbase1, coordbase2)
        e1, e2 = get_ellipticity(inertia_tensor)

        table_l['e_1'][i] = e1
        table_l['e_2'][i] = e2

    return table_l

path = '/home/hliu226/data/lensingdata/abacusLensing/lens_catalog'
data = Table.read(path + '/data_massive.fits')

table_l = calc_ellipticity(data)
table_l.write(path + '/lens_massive.fits', overwrite=True)

print('end of running')