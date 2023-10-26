import numpy as np
import torch
import scipy.sparse as sp
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import RPTree

from sklearn.neighbors import kneighbors_graph
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
# from normalization import fetch_normalization, row_normalize
from time import perf_counter


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)
    
def load_dataset1(normalization="AugNormAdj", cuda=False):
    """
    Load Citation Networks Datasets.
    """

def load_dataset(dataset_str, normalization, seed, flag_knn, flag_plot, NumOfTrees):
    """
    Loads input data from gcn/data directory

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    if dataset_str == 'iris':
        data = datasets.load_iris()
        features = data.data
        y = data.target
    elif dataset_str == 'wine':
        data = datasets.load_wine()
        features = data.data
        y = data.target
    elif dataset_str == 'BC-Wisc':
        data = datasets.load_breast_cancer()
        features = data.data
        y = data.target
    elif dataset_str == 'digits':
        data = datasets.load_digits()
        features = data.data
        y = data.target
    elif dataset_str == 'Olivetti':
        data = datasets.fetch_olivetti_faces()
        features = data.data
        y = data.target
    elif dataset_str == 'PenDigits':
        features = np.genfromtxt('data/PenDigits_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/PenDigits_Labels.csv', delimiter=",")
    elif dataset_str == 'mGamma':
        features = np.genfromtxt('data/mGamma_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/mGamma_Labels.csv', delimiter=",")
    elif dataset_str == 'CreditCard':
        features = np.genfromtxt('data/CreditCard_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/CreditCard_Labels.csv', delimiter=",")
    elif dataset_str == 'smile266':
        features = np.genfromtxt('data/smile266_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/smile266_Labels.csv', delimiter=",")
    elif dataset_str == 'Comp399':
        features = np.genfromtxt('data/Comp399_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/Comp399_Labels.csv', delimiter=",")
    elif dataset_str == 'Agg788':
        features = np.genfromtxt('data/Agg788_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/Agg788_Labels.csv', delimiter=",")
    elif dataset_str == 'sparse622':
        features = np.genfromtxt('data/sparse622_Instances.csv', delimiter=",")
        y = np.genfromtxt('data/sparse622_Labels.csv', delimiter=",")

    if flag_knn:
        # start knn adj matrix ---------------------------------------
        g = kneighbors_graph(features, 3, metric='minkowski')
        adj = g
        # end knn adj matrix ---------------------------------------
    else:
        # start rpTree adj matrix ---------------------------------------
        # NumOfTrees = n_tree
        adj1 = sp.coo_matrix(np.zeros((features.shape[0], features.shape[0]), dtype=np.float32))
        for r in range(NumOfTrees):
            tree = RPTree.BinaryTree(features)
            features_index = np.arange(features.shape[0])
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

            # convert edges list to adjacency matrix
            shape = tuple(edgeList.max(axis=0)[:2] + 1)
            adjMatRPTree = sp.coo_matrix((edgeList[:, 2], (edgeList[:, 0], edgeList[:, 1])), shape=shape,
                                             dtype=edgeList.dtype)

            # an adjacency matrix holding weights accumulated from all rpTrees
            adj1 = adj1 + (adjMatRPTree / NumOfTrees)
        # end rpTree adj matrix ---------------------------------------

    adj1 = adj1.toarray()
    labels = LabelBinarizer().fit_transform(y)
    if labels.shape[1] == 1:
        labels = np.hstack([labels, 1 - labels])
    n = features.shape[0]
    if dataset_str in ['sparse622', 'smile266', 'ring238', 'Agg788', 'iris', 'wine', 'BC-Wisc', 'digits']:
        n_train = 50
        n_val = 50
    else:
        n_train = n // 4
        n_val = n // 4
    idx_features = np.arange(len(y))
    from sklearn.model_selection import train_test_split
    train, test, y_train, y_test, idx_train, idx_test = train_test_split(features, y, idx_features, random_state=seed,
                                                    train_size=n_train + n_val,
                                                    test_size=n - n_train - n_val,
                                                    stratify=y)
    train, val, y_train, y_val, idx_train, idx_val = train_test_split(train, y_train, idx_train, random_state=seed,
                                                  train_size=n_train, test_size=n_val,
                                                  stratify=y_train)

    # get the indices of training samples and pair them
    idx_train_n = idx_train.size
    perm = np.empty((idx_train_n, idx_train_n, 2), dtype=idx_train.dtype)
    perm[..., 0] = idx_train[:, None]
    perm[..., 1] = idx_train
    perm1 = np.reshape(perm, (-1, 2))

    # retrieve the classes of the training samples and set the edge to -1 in case of different class
    # and 1 if they belong to the same class
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
    adj_idx_train = sp.coo_matrix((edgeList[:, 4], (edgeList[:, 0], edgeList[:, 1])), shape=(features.shape[0], features.shape[0]), dtype=edgeList.dtype)
    adj_idx_train.setdiag(0)
    if flag_plot:
        plot_adj_train(adj_idx_train.todense(), features, labels)

    adj_idx_train = adj_idx_train.toarray()
    adj = np.zeros_like(adj_idx_train)
    ablation_study = 'none'
    if ablation_study == 'intrinsic-graph':
        adj = adj_idx_train
        adj[adj < 0] = 0
        if flag_plot:
            plot_adj_histogram(adj)
        numofedges = np.count_nonzero(adj)//2
    elif ablation_study == 'penalty-graph':
        # add the edges from penalty graph to PCA tree edges
        adj = adj1
        adj_idx_train[adj_idx_train > 0] = 0
        adj = adj + adj_idx_train
        adj[adj < 0] = 0
        if flag_plot:
            plot_adj_histogram(adj)
        numofedges = np.count_nonzero(adj)//2
    elif ablation_study == 'PA-graph':
        adj = adj1
        if flag_plot:
            plot_adj_histogram(adj)
        numofedges = np.count_nonzero(adj)//2
    elif ablation_study == 'none':
        # add the edges from the training samples to PCA tree edges
        adj = adj1
        adj = adj + adj_idx_train
        adj[adj < 0] = 0
        adj[adj > 0] = 1
        if flag_plot:
            plot_adj_histogram(adj)
        # remove the edges that connect training samples from different classes
        if flag_plot:
            plot_adj_histogram(adj)
        numofedges = np.count_nonzero(adj)//2

    train_mask = np.zeros([n, ], dtype=bool)
    train_mask[idx_train] = True
    val_mask = np.zeros([n, ], dtype=bool)
    val_mask[idx_val] = True
    test_mask = np.zeros([n, ], dtype=bool)
    test_mask[idx_test] = True

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    if flag_plot:
        plot_true(adj.todense(), features, labels, idx_train, idx_val, idx_test)

    adj, features = preprocess_dataset(adj, features, dataset_str, normalization)

    features = sp.lil_matrix(features)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = torch.FloatTensor(adj)
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, numofedges

def preprocess_dataset(adj, features, dataset_str, normalization="FirstOrderGCN"):
    # identity matrix
    I = np.zeros_like(adj)
    np.fill_diagonal(I, 1)

    # degree matrix
    D = np.zeros_like(adj)
    np.fill_diagonal(D, np.sum(adj,axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D+I))

    adj_hat = np.dot(D_inv_sqrt, adj+I).dot(D_inv_sqrt)

    # adj_normalizer = fetch_normalization(normalization)
    # adj = adj_normalizer(adj)
    if dataset_str == 'iris' or dataset_str == 'wine':
        features = row_normalize(features)
    return adj_hat, features

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.mm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def plot_true(adj, features, labels, idx_train, idx_val, idx_test):
    # plot adjacency matrix
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    G_1 = nx.from_numpy_array(adj)
    nx.draw(G_1, features, node_size=20, alpha=0.75)
    date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    #plt.savefig(date_string+'.png', dpi=150, bbox_inches='tight')
    plt.savefig(date_string + '-adj.png', bbox_inches='tight')
    plt.close()

    y = np.argmax(labels, axis=1)
    plot_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    #plot all samples
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for r in range(features.shape[0]):
        ax.scatter(features[r, 0], features[r, 1], color=plot_color[y[r]])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.savefig(date_string + '-features.png', bbox_inches='tight')
    plot_x_axis = ax.get_xlim()
    plot_y_axis = ax.get_ylim()
    plt.close()

    #plot training samples
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for r in idx_train:
        ax.scatter(features[r, 0], features[r, 1], color=plot_color[y[r]])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.set_xlim(plot_x_axis)
    ax.set_ylim(plot_y_axis)
    plt.savefig(date_string + '-training.png', bbox_inches='tight')
    plt.close()

    #plot validation samples
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for r in idx_val:
        ax.scatter(features[r, 0], features[r, 1], color=plot_color[y[r]])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.set_xlim(plot_x_axis)
    ax.set_ylim(plot_y_axis)
    plt.savefig(date_string + '-validation.png', bbox_inches='tight')
    plt.close()

    #plot testing samples with true labels
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for r in idx_test:
        ax.scatter(features[r, 0], features[r, 1], color=plot_color[y[r]])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.set_xlim(plot_x_axis)
    ax.set_ylim(plot_y_axis)
    plt.savefig(date_string + '-testing-true.png', bbox_inches='tight')
    plt.close()

def plot_features_propagation(features, labels, degree):
    # convert to numpy arrays
    features = features.numpy()
    labels = labels.numpy()

    plot_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    #plot all samples
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for r in range(features.shape[0]):
        ax.scatter(features[r, 0], features[r, 1], color=plot_color[labels[r]])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    plt.savefig(date_string+'-features-propagation-degree-'+str(degree)+'.png', bbox_inches='tight')
    plt.close()


def plot_predicted(output, labels, test_features):
    # compute the predicted labels from the output features
    predicted_labels = output.max(1)[1].type_as(labels)
    # convert to numpy arrays
    predicted_labels = predicted_labels.numpy()
    test_features = test_features.numpy()

    plot_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


    #plot testing samples with predicted labels
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for r in range(test_features.shape[0]):
        ax.scatter(test_features[r, 0], test_features[r, 1], color=plot_color[predicted_labels[r]])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    # ax.set_xlim(plot_x_axis)
    # ax.set_ylim(plot_y_axis)
    date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    plt.savefig(date_string + '-testing-predicted.png', bbox_inches='tight')
    plt.close()


def plot_adj_train(adj, features, labels):
    y = np.argmax(labels, axis=1)
    plot_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plot_color_node = []
    for i in range(len(y)):
        plot_color_node.append(plot_color[y[i]])

    # plot adjacency matrix
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    G_1 = nx.from_numpy_array(adj)

    plot_color_edge = []
    for (u, v, c) in G_1.edges.data('weight'):
        if c < 1:
            plot_color_edge.append('r')
        else:
            plot_color_edge.append('k')

    nx.draw(G_1, features, node_color=plot_color_node, edge_color=plot_color_edge, node_size=20, alpha=0.5)
    date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    #plt.savefig(date_string+'.png', dpi=150, bbox_inches='tight')
    plt.savefig(date_string + '-adj-train.png', bbox_inches='tight')
    plt.close()


def plot_adj_histogram(adj):
    numofedges = adj.nnz // 2
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    plt.hist(adj.toarray().flatten(), density=False)  # density=False would make counts
    plt.title('Number of edges = ' + str(numofedges))
    plt.ylabel('counts')
    plt.xlabel('Data')
    date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")
    plt.savefig(date_string + '-adj-histogram.png', bbox_inches='tight')
    plt.close()
