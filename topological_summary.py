#!/usr/bin/env python

from models.utils import get_model
import torch
import torch.nn as nn
from loaders import *
from graph import *
from utils import save_dipha
from passers import Passer
from config import SAVE_PATH, MAX_EPSILON, UPPER_DIM, NPROC
from pathlib import Path
from bettis import read_pd
import numpy as np
import pickle
from pathlib import Path
import multiprocessing


def calc_adjacency_matrix(activations, verbose=False):
    """ Build matrix A of dimensions nxn where A_{ij} = pearson's correlation(a_i, a_j)
    
    Parameters
    ----------
    activations: list
        List of ndarray containing activation signals. Position in list
        corresponds to the layer (with activations, e.g. no pooling) in network. 
        Each ndarray has shape (samples, depth, width, height). 
    verbose: bool, optional
        Print verbose info (default: False).
   
    Returns
    -------
    ndarray
        a nxn matrix containng pearson's correlations between activations
    """
    activations = signal_concat(activations)
    adj = adjacency(activations)

    if verbose:
        print('The dimension of the adjacency matrix is {}.'.format(adj.shape))
        print('Adj mean {}, min {}, max {}'.format(np.mean(adj), np.min(adj), np.max(adj)))

    return adj

def compute_persistent_homology(export_dir, adjacency_file, sparsemat_file, pershomology_file,  
        max_epsilon, upper_dim, nproc, verbose=False):
    """ Filters adjacency matrix and computes persistence homology from it.

    Parameters
    ----------
    exportdir: str, Path
        Directory where temporary files are stored.
    adjacency_file: str
        Name of intermediate file where adjacency matrix is stored in.
    sparsemat_file: str
        Name of intermediate file where sparse adjacency matrix gets stored in.
    pershomology_file: str
        Name of intermediate file where persistent homology gets stored in.
    max_epsilon: float
        Upper limit for distance between nodes where they still can be connected
        to a simplex.
    upper_dim: int
        Highest dimension for which persistence homology is computed.
    nproc: int
        Number of CPU cores that can be used for processing.
    verbose: bool, optional
        Print verbose info (default: False).

    Returns
    -------
    """
    export_dir = Path(export_dir)

    adjacency_export_file = export_dir / adjacency_file
    sparsemat_export_file = export_dir / sparsemat_file
    pershomology_export_file = export_dir / pershomology_file
    if verbose:
        print(f"Adjacency matrix file: {adjacency_export_file.absolute()}")
        print(f"Sparse adjacency matrix file: {sparsemat_export_file.absolute()}")
        print(f"Persistent homology export file: {pershomology_export_file.absolute()}")
        print(f"Max Epsilon: {max_epsilon}")
        print(f"Upper Dimension: {upper_dim}")
        print(f"Number of used processor cores: {nproc}")

    # create sparse matrix by cuttin off all values above 'max_epsilon' (?)
    os.system("./dipha/build/full_to_sparse_distance_matrix "
            + str(max_epsilon) + " " +  str(adjacency_export_file.absolute()) + " " + 
            str(sparsemat_export_file.absolute()))

    os.system("mpiexec -n " +str(nproc) + " ./dipha/build/dipha --upper_dim "
            + str(upper_dim) + " --benchmark --dual " + str(sparsemat_export_file.absolute()) + " "
            + str(pershomology_export_file.absolute()))

def calc_summary(export_dir, pershomology_file, dim, persistence, verbose=False):
    """ Calculates topological summary from given persistent homology

    Parameters
    ----------
    exportdir: str
        Directory where temporary files are stored in.
    pershomology_file: str
        Name of intermediate file where persistent homology is stored in.
    dim: int
        Dimension for which topological summary is returned
    persistence: float
        ?
    verbose: bool, optional
        Print verbose information (default: False).

    Returns
    -------
    (my, lambda): (float, float)
        Topological summary with 'my' being average life and 'lambda' being midlife 
        of cavities in topology
    """

    export_dir = Path(export_dir)
    pershomology_export_file = export_dir / pershomology_file

    birth, death = np.array(
            read_pd(pershomology_export_file, dimension=dim, persistence=persistence)
            )
    avg_life = np.mean(death - birth)
    midlife = np.mean((birth + death)/2)
    if verbose:
        print("Average Life: {}".format(avg_life))
        print("Midlife: {}".format(midlife))

    return (avg_life, midlife)


def compute_topological_summary(activations, export_dir="/tmp/", adjacency_file='adj.bin', 
        sparsemat_file='sparsemat.out', pershomology_file='ph.out',
        max_epsilon=0.3, upper_dim=1, nproc=None, dim=0, persistence=0.02,
        verbose=False):
    """ Compute topological summary for topology of network models
    functional space given by sample activations.

    Parameters
    ----------
    activations: list
        List of ndarray containing activation signals. Position in list
        corresponds to the layer (with activations, e.g. no pooling) in network. 
        Each ndarray has shape (samples, depth, width, height). 
    export_dir: str, optional
        Directory to store intermediate files in (default is '/tmp/').
    adjacency_file: str, optional
        Name of intermediate file to store adjacency matrix in. DIPHA algorithm
        later uses these files (default: 'adj.bin').
    sparsemat_file: str, optional
        Name of intermediate file where sparse adjacency matrix gets stored in 
        (default: 'sparsemat.out').
    pershomology_file: str, optional
        Name of intermediate file where persistent homology gets stored in (default: 'ph.out').
    max_epsilon: float, optional
        upper limit for distance between nodes where they still can be connected
        to a simplex (default: 0.3).
    upper_dim: int, optional
        Highest dimension for which persistence homology is computed (default: 1).
    nproc: int, optinal
        Number of CPU cores that can be used for processing (default: all available, max: 16).
    dim: int, optional
        Dimension for which topological summary is returned (default: 0).
    persistence: float, optional
        ? (default: 0.02)

    Returns
    -------
    (my, lambda): (float, float)
        Topological summary with 'my' being average life and 'lambda' being midlife 
        of cavities in topology.

    """
    if(nproc is None):
        nproc = min(multiprocessing.cpu_count(), 16)
        
    export_dir = Path(export_dir)
    adjacency_export_path = export_dir / adjacency_file

    adj = calc_adjacency_matrix(activations=activations, verbose=verbose)

    # Write adjacency matrix to binary to use DIPHA 
    # input for persistence homology.
    save_dipha(adjacency_export_path, 1-adj)
    compute_persistent_homology(export_dir=export_dir, adjacency_file=adjacency_file,
            sparsemat_file=sparsemat_file, pershomology_file=pershomology_file,
            max_epsilon=max_epsilon, upper_dim=upper_dim, nproc=nproc, verbose=verbose)

    return calc_summary(export_dir=export_dir, pershomology_file=pershomology_file, 
            dim=dim, persistence=persistence, verbose=verbose)

if __name__ == '__main__':
    with open('../../alexnet_activations.pkl', 'rb') as f:
        data = pickle.load(f)
        #activations = [activation.detach().numpy() for activation in data]
        activations = data
        compute_topological_summary(activations, verbose=True)
