import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree

import RPTree

def build_PATree_graph(dataset):
    tree = RPTree.BinaryTree(dataset['features'].numpy())
    features_index = np.arange(dataset['features'].shape[0])
    tree_root = tree.construct_tree(tree, features_index)

    # get the indices of points in leaves
    leaves_array = tree_root.get_leaf_nodes()

    # connect points in the same leaf node
    edgeList = []
    for i in range(len(leaves_array)):
        x = leaves_array[i]
        n = x.size
        perm = np.empty((n, n, 2), dtype=x.dtype)
        perm[..., 0] = x[:, None]
        perm[..., 1] = x
        perm1 = np.reshape(perm, (-1, 2))
        if i == 0:
            edgeList = perm1
        else:
            edgeList = np.vstack((edgeList, perm1))

    # assign one as edge weight
    edgeList = edgeList[edgeList[:, 0] != edgeList[:, 1]]
    edgeList = np.hstack((edgeList, np.ones((edgeList.shape[0], 1), dtype=int)))

    # convert edges list to an adjacency matrix
    adjMatPATree = sp.coo_matrix((edgeList[:, 2], (edgeList[:, 0], edgeList[:, 1])), 
                                shape=(dataset['features'].shape[0], dataset['features'].shape[0]),
                                dtype=edgeList.dtype)

    adjMatPATree = adjMatPATree.toarray()

    return adjMatPATree

def add_intrinsic_graph(dataset, adjMatPATree):
    # get the indices of training samples and pair them
    idx_train = np.nonzero(dataset['train_mask'].numpy())[0]
    idx_train_n = idx_train.size
    perm = np.empty((idx_train_n, idx_train_n, 2), dtype=idx_train.dtype)
    perm[..., 0] = idx_train[:, None]
    perm[..., 1] = idx_train
    perm1 = np.reshape(perm, (-1, 2))

    # retrieve the classes of the training samples and set the edge to -1 in case of different class
    # and 1 if they belong to the same class
    y = dataset['labels'].numpy()
    v1_class = y[perm1[:, 0]]
    v1_class = v1_class[..., None]
    v2_class = y[perm1[:, 1]]
    v2_class = v2_class[..., None]
    idx_train_labels = np.hstack((v1_class, v2_class))
    edgeList = np.hstack((perm1, idx_train_labels))
    idx_train_match = np.zeros((len(v1_class),), dtype=int)
    for i in range(len(v1_class)):
        if edgeList[i, 2] == edgeList[i, 3]:
            # set this to zero to ignore intrinsic graph edges
            idx_train_match[i] = 1
        else:
            idx_train_match[i] = -1

    idx_train_match = idx_train_match[..., None]
    edgeList = np.hstack((edgeList, idx_train_match))
    adj_idx_train = sp.coo_matrix((edgeList[:, 4], (edgeList[:, 0], edgeList[:, 1])), shape=(dataset['features'].shape[0], dataset['features'].shape[0]), dtype=edgeList.dtype)
    adj_idx_train.setdiag(0)
    adj_idx_train = adj_idx_train.toarray()

    adj = adjMatPATree
    # add the edges from the training samples to PCA tree edges
    adj = adj + adj_idx_train
    # remove the edges that connect training samples from different classes
    adj[adj < 0] = 0
    adj[adj > 0] = 1

    return adjMatPATree


# Load the CiteSeer dataset
dataset_name = 'CiteSeer' #'Cora' 'CiteSeer'
dataset_raw = Planetoid('/kaggle/working/',name=dataset_name)

dataset = dict()
dataset['edgeslist'] = dataset_raw[0].edge_index.t()
dataset['features'] = dataset_raw[0].x
dataset['labels'] = dataset_raw[0].y
dataset['train_mask'] = dataset_raw[0].train_mask
dataset['val_mask'] = dataset_raw[0].val_mask
dataset['test_mask'] = dataset_raw[0].test_mask

adj_dataset = sp.coo_matrix((np.ones((dataset['edgeslist'].shape[0],1)).squeeze(),
                              (dataset['edgeslist'][:, 0].numpy().squeeze(),
                               dataset['edgeslist'][:, 1].numpy().squeeze())),
                             shape=(dataset['features'].shape[0], dataset['features'].shape[0]))
adj_dataset = adj_dataset.toarray()


for runs in range(1):

    #####################################################
    #   Construct PA-tree graph                         #
    #####################################################
    adjMatPATree = build_PATree_graph(dataset)
    adj = add_intrinsic_graph(dataset, adjMatPATree)

    print('Number of edges in adj = '+str(np.count_nonzero(adj)))
    # plt.matshow(adj)
    # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
    #                 labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    # date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    # plt.savefig(date_string + '-adj-compare.png', bbox_inches='tight')
    # plt.close()

    adj_compare = np.zeros((2,2))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if (adj[i,j]==0) & (adj_dataset[i,j]==0):
                adj_compare[0,0] += 1
            elif (adj[i,j]==1) & (adj_dataset[i,j]==0):
                adj_compare[1,0] += 1
            elif (adj[i,j]==0) & (adj_dataset[i,j]==1):
                adj_compare[0,1] += 1
            elif (adj[i,j]==1) & (adj_dataset[i,j]==1):
                adj_compare[1,1] += 1

    with open('Results-' + dataset_name + '.csv', 'a') as my_file:
        # Dataset, graph type, adj_compare[0,0], adj_compare[1,0], adj_compare[0,1], adj_compare[1,1]
        my_file.write('\n')
        my_file.write(dataset_name + ',' + 'PA-tree' + ','
                      + str(np.round(adj_compare[0,0]/np.sum(adj_compare), 4)) + ',' 
                      + str(np.round(adj_compare[1,0]/np.sum(adj_compare), 4)) + ','
                      + str(np.round(adj_compare[0,1]/np.sum(adj_compare), 4)) + ','
                      + str(np.round(adj_compare[1,1]/np.sum(adj_compare), 4)) + ',')
        
    #####################################################
    #   Construct $ϵ$-graph                             #
    #####################################################

    Tcsr = minimum_spanning_tree(pairwise_distances(dataset['features'],dataset['features']))
    epsilon_value = np.max(Tcsr.toarray())
    adj_epsilon = radius_neighbors_graph(dataset['features'], epsilon_value)
    adj_epsilon = adj_epsilon.toarray()

    print('Number of edges in adj_epsilon = '+str(np.count_nonzero(adj_epsilon)))

    adj_compare = np.zeros((2,2))
    for i in range(adj_epsilon.shape[0]):
        for j in range(adj_epsilon.shape[1]):
            if (adj_epsilon[i,j]==0) & (adj_dataset[i,j]==0):
                adj_compare[0,0] += 1
            elif (adj_epsilon[i,j]==1) & (adj_dataset[i,j]==0):
                adj_compare[1,0] += 1
            elif (adj_epsilon[i,j]==0) & (adj_dataset[i,j]==1):
                adj_compare[0,1] += 1
            elif (adj_epsilon[i,j]==1) & (adj_dataset[i,j]==1):
                adj_compare[1,1] += 1

    with open('Results-' + dataset_name + '.csv', 'a') as my_file:
        # Dataset, graph type, adj_compare[0,0], adj_compare[1,0], adj_compare[0,1], adj_compare[1,1]
        my_file.write('\n')
        my_file.write(dataset_name + ',' + '$\epsilon$' + ','
                        + str(np.round(adj_compare[0,0]/np.sum(adj_compare), 4)) + ',' 
                        + str(np.round(adj_compare[1,0]/np.sum(adj_compare), 4)) + ','
                        + str(np.round(adj_compare[0,1]/np.sum(adj_compare), 4)) + ','
                        + str(np.round(adj_compare[1,1]/np.sum(adj_compare), 4)) + ',')
        
    #####################################################
    #   Construct $k$-nn graph                          #
    #####################################################

    k_value = np.log(dataset['features'].shape[0]).astype(int)
    adj_k = kneighbors_graph(dataset['features'], k_value)
    adj_k = adj_k.toarray() 

    print('Number of edges in adj_k = '+str(np.count_nonzero(adj_k)))

    adj_compare = np.zeros((2,2))
    for i in range(adj_k.shape[0]):
        for j in range(adj_k.shape[1]):
            if (adj_k[i,j]==0) & (adj_dataset[i,j]==0):
                adj_compare[0,0] += 1
            elif (adj_k[i,j]==1) & (adj_dataset[i,j]==0):
                adj_compare[1,0] += 1
            elif (adj_k[i,j]==0) & (adj_dataset[i,j]==1):
                adj_compare[0,1] += 1
            elif (adj_k[i,j]==1) & (adj_dataset[i,j]==1):
                adj_compare[1,1] += 1

    with open('Results-' + dataset_name + '.csv', 'a') as my_file:
        # Dataset, graph type, adj_compare[0,0], adj_compare[1,0], adj_compare[0,1], adj_compare[1,1]
        my_file.write('\n')
        my_file.write(dataset_name + ',' + '$k$-nn' + ','
                        + str(np.round(adj_compare[0,0]/np.sum(adj_compare), 4)) + ',' 
                        + str(np.round(adj_compare[1,0]/np.sum(adj_compare), 4)) + ','
                        + str(np.round(adj_compare[0,1]/np.sum(adj_compare), 4)) + ','
                        + str(np.round(adj_compare[1,1]/np.sum(adj_compare), 4)) + ',')                
        