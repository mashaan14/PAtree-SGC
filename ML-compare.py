import numpy as np

from sklearn import datasets as skdatasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset(dataset_str):
    if dataset_str == 'iris':
        data = skdatasets.load_iris()
        features = data.data
        y = data.target
    elif dataset_str == 'wine':
        data = skdatasets.load_wine()
        features = data.data
        y = data.target
    elif dataset_str == 'BC-Wisc':
        data = skdatasets.load_breast_cancer()
        features = data.data
        y = data.target
    elif dataset_str == 'digits':
        data = skdatasets.load_digits()
        features = data.data
        y = data.target
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

    return features, y

seed = 42
datasets = ['smile266', 'Comp399', 'sparse622', 'Agg788', 'iris', 'wine', 'BC-Wisc', 'digits']
names = [
    "Nearest Neighbors",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",

]

classifiers  = (
    KNeighborsClassifier(5),
    SVC(gamma=2, C=1, random_state=seed),
    DecisionTreeClassifier(max_depth=5, random_state=seed),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=seed
    ),
)

for ds in datasets:
  # preprocess dataset, split into training and test part
  X, y = load_dataset(ds)
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=50, random_state=seed)
  # iterate over classifiers
  for name, clf in zip(names, classifiers):
    for runs in range(10):
      clf = make_pipeline(StandardScaler(), clf)
      clf.fit(X_train, y_train)
      score = clf.score(X_test, y_test)
      # print(ds+ ' ' + name + ' ' + str(np.round(score, 4)))
      with open('Results' + '.csv', 'a') as my_file:
        # Dataset, Method, Test accuracy
        my_file.write('\n')
        my_file.write(ds + ',' + name + ',' + str(np.round(score, 4)))
