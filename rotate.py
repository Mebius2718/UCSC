import numpy as np
import astropy.units as u
from tqdm import tqdm
from astropy.table import Table

def rotation_matrix(matrix1, matrix2):
    return np.dot(matrix2.T, matrix1)

def inertia_tensor_transform(inertia_tensor, coord_base1, coord_base2):
    """
    Transforms the inertia tensor from one coordinate basis to another.

    inertia_tensor: 
        3x3 matrix of inertia tensor
    coord_base1: 
        3x3 matrix of the original coordinate basis
    coord_base2:
        3x3 matrix of the new coordinate basis
    
    Returns:
        3x3 matrix of the transformed inertia tensor
    """
    
    R = rotation_matrix(coord_base1, coord_base2)

    return R @ inertia_tensor @ R.T

def get_inertia_tensor(eigenvalues):
    """
    Finds the inertia tensor from eigenvalues and eigenvectors.
    """
    return np.diag(eigenvalues**2)

def get_coordbase_parallel_ra_dec(ra, dec):
    """
    Finds the coordinate basis from eigenvectors.
    
    A very strightforward diagram: https://arxiv.org/pdf/2309.08605 Sect.2.3

    ra: rad
        Right ascension of the galaxy
    dec: rad
        Declination of the galaxy

    Returns:
        3x3 matrix of the coordinate basis which is parallel to the ra, dec, and los
    """

    # ra, dec to theta, phi in spherical coordinates
    phi1 = np.deg2rad(90 - dec)
    phi2 = np.deg2rad(ra)
    # phi2 = np.deg2rad(90 + ra)

    cos_phi1 = np.cos(phi1)
    sin_phi1 = np.sin(phi1)
    cos_phi2 = np.cos(phi2)
    sin_phi2 = np.sin(phi2)

    vec_phi1 = np.array([cos_phi1*cos_phi2, cos_phi1*sin_phi2, -sin_phi1])
    vec_phi2 = np.array([-sin_phi2, cos_phi2, 0])
    vec_n = -np.array([sin_phi1*cos_phi2, sin_phi1*sin_phi2, cos_phi1])

    x1 = vec_phi1
    x2 = vec_phi2
    x3 = -vec_n

    coordbase = np.array([x1, x2, x3]).T

    return coordbase

def schur_complement(mat):
    """
    Computes the Schur complement of a 3x3 matrix
    """
    topLeft2x2 = mat[:2,:2]
    right1x2 = mat[2:,:2]
    bottom1x1 = mat[2:,2:]
    assert bottom1x1.shape == (1,1), "Matrix must be 3x3"
    return topLeft2x2 - 1/bottom1x1 * right1x2.T @ right1x2

def get_ellipticity(mat):
    if len(mat) == 3:
        mat = schur_complement(mat)
    det = np.linalg.det(mat)
    e1 = (mat[0, 0] - mat[1, 1]) / (mat[0, 0] + mat[1, 1] + 2 * np.sqrt(det))
    e2 = (2 * mat[0, 1]) / (mat[0, 0] + mat[1, 1] + 2 * np.sqrt(det))
    return e1, e2