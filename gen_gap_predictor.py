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

class GenGapPredictor:
    def __init__(self, net, dataloader):
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net
        self.dataloader = dataloader
        self.net.to(self.device) # move net to gpu if necessary
        self.net = torch.nn.DataParallel(net) # allow usage of multiple gpus
        self.criterion = nn.CrossEntropyLoss()

    def _construct_functional_graph(self, epoch, checkpoint):
        print("="*50)
        print("Construction functional graph for model:")
        print(net)
        print("="*50)
        print("==> Loading checkpoint for epoch {} ...".format(epoch))
        self.net.load_state_dict(checkpoint['net'])
        passer_test = Passer(self.net, self.dataloader, self.criterion, self.device)
        print()
        passer_test.run()
        activations = passer_test.get_function()
        activations = signal_concat(activations)
        adj = adjacency(activations)
        print('The dimension of the adjacency matrix is {}.'.format(adj.shape))
        print('Adj mean {}, min {}, max {}'.format(np.mean(adj), np.min(adj), np.max(adj)))

        return adj

    def export_functional_graph_dipha(self, adj, save_dir, epoch):
        # save functional graph to binary file to load it later with DIPHA library.
        graphfile_path = save_dir + 'adj_epc{}_trl0.bin'.format(epoch)
        print("Saving functional graph to '"+graphfile_path+"' ...")
        save_dipha(graphfile_path, 1-adj)
        return graphfile_path

    def compute_topology(self, graph_file_path):
        print(graphfile_path)

        # get epoch from path string
        epoch = int(graphfile_path.split('/')[-1].split('_')[1].replace('epc',' '))
        
        # construct path to store vietoris-rips filtration in
        filtration_path = os.path.join(Path(graphfile_path).parent , 'adj_epc{}_trl0_{}.bin'.format(
            epoch, MAX_EPSILON))

        # construct path to store results in
        persistence_diag_path = filtration_path + ".out"

        # calculate vietoris-rips filtration of function graph and 
        # store it in 'out_path'
        os.system("./dipha/build/full_to_sparse_distance_matrix "
                + str(MAX_EPSILON) + " " + graphfile_path + " "+ filtration_path)

        # calculate persistence diagram
        os.system("mpiexec -n " +str(NPROC) + " ./dipha/build/dipha --upper_dim " 
                + str(UPPER_DIM) + " --benchmark --dual " + filtration_path + " "
                + persistence_diag_path)

        return persistence_diag_path

    def calculate_summary(self, persistence_diag_path, dim=1, persistence=0.02):
        birth, death = np.array(
                read_pd(persistence_diag_path, dimension=dim, persistence=persistence)
                )
        avg_life = np.mean(death - birth)
        midlife = np.mean((birth + death)/2)
        print("Average Life: {}".format(avg_life))
        print("Midlife: {}".format(midlife))

        return avg_life, midlife


if __name__ == '__main__':
    model = 'lenet'
    dataset = 'mnist'
    epoch = 0
    save_dir = os.path.join(SAVE_PATH, model + '_' + dataset + '/')

    checkpoint = torch.load(
        './checkpoint/'+ model + '_' + dataset + '/ckpt_trial_0_epoch_' + str(epoch)+'.t7'
            )

    dataloader = loader(dataset+'_test', batch_size=100, subset=list(range(0,1000)))

    net = get_model(model, dataset)
    predictor = GenGapPredictor(net, dataloader)
    funcgraph = predictor._construct_functional_graph(epoch, checkpoint)
    graphfile_path = predictor.export_functional_graph_dipha(funcgraph, save_dir, epoch)
    persistence_diag_path = predictor.compute_topology(graphfile_path)
    avg_life, midlife = predictor.calculate_summary(persistence_diag_path)
