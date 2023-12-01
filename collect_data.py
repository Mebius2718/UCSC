import asdf, os, sys
import numpy as np
import healpy as hp
import scipy.stats as scist
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from tqdm import trange,tqdm
from astropy.table import Table,hstack,vstack,join

def load_catalog(path, fields_lc):
    data = asdf.open(path)['data']
    cat = Table()
    for key in fields_lc:
        column = data[key]
        if isinstance(column, dict):
            cat[key] = column['data']
        else:
            cat[key] = column
    return cat

def load_lightcone_halos(z, los,
        basepath = '/data/groups/leauthaud/sven/AbacusMocks/',
        fields_lc=['RA','DEC','Z_COSMO']):

    fn = basepath+os.sep+f"AbacusSummit_base_c000_ph{los:03d}/z{z:.3f}/catalog_halos.asdf"
    return load_catalog(fn, fields_lc)

def load_halo_masses(z, los,
        basepath = '/data/groups/leauthaud/sven/AbacusMocks/',
        fields_lc = ['MassHMsun']):

    fn = basepath + f"AbacusSummit_base_c000_ph{los:03d}/z{z:.3f}/mass_halos.asdf"
    return load_catalog(fn, fields_lc)

def load_halo_properties(z, los,
        basepath = '/data/groups/leauthaud/sven/AbacusMocks/',
        fields_lc = ['sigman_eigenvecsMin_L2com',
                     'sigman_eigenvecsMid_L2com',
                     'sigman_eigenvecsMaj_L2com',
                     'sigman_L2com',
                     'pos_interp']):

    fn = basepath + f"AbacusSummit_base_c000_ph{los:03d}/z{z:.3f}/lc_halo_info.fits"
    data = Table.read(fn)
    data = data[fields_lc]

    return data

def match_halos(z, los, mass_cut = None):
    lc_halos = load_lightcone_halos(z,los)
    halo_props = load_halo_properties(z,los)
    halo_mass = load_halo_masses(z,los)
    if mass_cut is not None:
        mask = halo_mass['MassHMsun'] > mass_cut
        lc_halos = lc_halos[mask]
        halo_props = halo_props[mask]
        halo_mass = halo_mass[mask]
    tab = hstack([lc_halos, halo_props, halo_mass])
    mask_pos = np.all(np.isclose(tab['pos_interp'],np.zeros(3)),axis=1)
    if(np.any(mask_pos)):
        print(f"Found and removed {mask_pos.sum()} halos with zero position.")
        tab = tab[~mask_pos]
    return tab

def build_galaxy_catalogue(los,z_max=2.6,mass_cut=None,
    # z_array=np.array([0.100,0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500,0.575,0.650,0.725,0.800,0.875,0.950,1.025,1.100,1.175,1.250,1.325,1.400,1.475,1.550,1.625,1.700,1.850,2.000,2.250,2.500]),
        z_array = np.array([0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500,0.575,0.650,0.725,0.800,0.875,0.950,1.025,1.100,1.175,1.250,1.325,1.400,1.475,1.550,1.625,1.700,1.850,2.000]),
        origin = np.array([-990,-990,-990])):

    tab = Table()
    z_array_masked = z_array[z_array<z_max]
    for z in tqdm(z_array_masked,desc='Assembling catalogue'):
        tab_z = match_halos(z,los,mass_cut=mass_cut)
        tab = vstack([tab,tab_z],metadata_conflicts='silent')
    return tab

data = build_galaxy_catalogue(los=0,z_max=1.1,mass_cut=1e11)
data.write('/home/hliu226/data/lensingdata/abacusLensing/lens_data.fits',overwrite=True)