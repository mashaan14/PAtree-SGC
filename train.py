import argparse
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from models import get_model
from metrics import accuracy

# # Args SGC
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default="sparse622", choices=['sparse622', 'smile266', 'ring238', 'Agg788',
#                                                                         'iris', 'wine', 'BC-Wisc', 'digits',
#                                                                         'Olivetti', 'PenDigits', 'mGamma', 'CreditCard'], help='Dataset to use.')
# parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
# parser.add_argument('--inductive', action='store_true', default=False, help='inductive training.')
# parser.add_argument('--test', action='store_true', default=False, help='inductive training.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-6, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=0, help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--normalization', type=str, default='AugNormAdj', choices=['AugNormAdj'], help='Normalization method for the adjacency matrix.')
# parser.add_argument('--model', type=str, default="SGC", help='model to use.')
# parser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
# parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')

# Args GCN
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="sparse622", choices=['sparse622', 'smile266', 'ring238', 'Agg788',
                                                                        'iris', 'wine', 'BC-Wisc', 'digits',
                                                                        'Olivetti', 'PenDigits', 'mGamma', 'CreditCard'], help='Dataset to use.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--inductive', action='store_true', default=False, help='inductive training.')
parser.add_argument('--test', action='store_true', default=False, help='inductive training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--normalization', type=str, default='AugNormAdj', choices=['AugNormAdj'], help='Normalization method for the adjacency matrix.')
parser.add_argument('--model', type=str, default="GCN", help='model to use.')
parser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed, args.cuda)

for runs in range(10):

    adj, features, labels, idx_train, idx_val, idx_test, numofedges = load_dataset(args.dataset, args.normalization, args.seed, flag_knn=False, flag_plot=False)

    model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

    if args.model == "SGC":
        features, precompute_time = sgc_precompute(features, adj, args.degree)
    else:
        precompute_time = 0
    print("{:.4f}s".format(precompute_time))
    # plot_features_propagation(features, labels, args.degree)


    def train_regression(model,
                         train_features, train_labels,
                         val_features, val_labels,
                         epochs=args.epochs, weight_decay=args.weight_decay,
                         lr=args.lr, dropout=args.dropout):

        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        t = perf_counter()
        for epoch in range(epochs):
            # sets the mode to training mode
            model.train()
            optimizer.zero_grad()
            # run the model with the training samples and return the output
            output = model(train_features)
            # compute the cross entropy loss between the training output and the true labels.
            loss_train = F.cross_entropy(output, train_labels)
            # calculate the gradients of the loss function with respect to the parameters
            loss_train.backward()
            # adjust the weights (optimize) using the gradients.
            optimizer.step()
        train_time = perf_counter()-t

        with torch.no_grad():
            model.eval()
            output = model(val_features)
            acc_val = accuracy(output, val_labels)

        return model, acc_val, train_time


    def test_regression(model, test_features, test_labels, flag_plot):
        model.eval()
        if flag_plot:
            plot_predicted(model(test_features), test_labels, test_features)
        return accuracy(model(test_features), test_labels)


    def train(epoch):
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        t = perf_counter()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(perf_counter() - t))
        return model, acc_val


    def test(model, test_features, test_labels, flag_plot):
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "accuracy= {:.4f}".format(acc_test.item()))
        if flag_plot:
            plot_predicted(output, test_labels, test_features)
        return acc_test


    if args.model == "SGC":
        model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val],
                                                      labels[idx_val], args.epochs, args.weight_decay,
                                                      args.lr, args.dropout)
        acc_test = test_regression(model, features[idx_test], labels[idx_test], flag_plot=False)
        print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
        print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total time: {:.4f}s".format(precompute_time, train_time,
                                                                                           precompute_time + train_time))
    elif args.model == "GCN":
        t = perf_counter()
        for epoch in range(args.epochs):
            model, acc_val = train(epoch)
        train_time = perf_counter() - t
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(train_time))

        acc_test = test(model, features[idx_test], labels[idx_test], flag_plot=False)

    with open('Results-' + args.dataset + '.csv', 'a') as my_file:
        # Dataset, Method, Validation accuracy, Test accuracy, Number of edges, Pre-compute time, train time, total time
        my_file.write('\n')
        my_file.write(args.dataset + ',' + args.model + ',' + str(np.round(acc_val.numpy(), 4)) + ',' +
                      str(np.round(acc_test.numpy(), 4)) + ',' + str(numofedges) + ',' +
                      str(np.round(precompute_time, 4)) + ',' + str(np.round(train_time, 4)) + ',' +
                      str(np.round(precompute_time+train_time, 4)))
