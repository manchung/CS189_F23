from collections import Counter

import io
import numpy as np
from numpy import genfromtxt
import pandas as pd
from pydot import graph_from_dot_data
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO implement entropy function
        # pass
        y_values, y_value_counts = np.unique(y, return_counts=True)
        probs = y_value_counts / y_value_counts.sum()
        log_probs = np.log(probs)
        return -1*(probs * log_probs).sum()

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        # return np.random.rand()
        lower_idx = X<thresh
        y_lower = y[lower_idx]
        y_upper = y[~lower_idx]

        ig = DecisionTree.entropy(y) - (
            len(y_lower) * DecisionTree.entropy(y_lower) + len(y_upper) * DecisionTree.entropy(y_upper)
            ) / len(y)
        
        return ig


    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(
                    np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([
                    self.information_gain(X[:, i], y, t) for t in thresh[i, :]
                ])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(
                np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(
                X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(
                X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, n=200, params=None):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        # pass
        for tree in self.decision_trees:
            rand_idx = np.random.randint(low=0, high=len(y), size=len(y))
            tree.fit(X[rand_idx, :], y[rand_idx])

    def predict(self, X):
        # TODO implement function
        tree_predicts = np.zeros((X.shape[0], self.n))
        yhat = np.zeros(X.shape[0])
        for i in range(self.n):
            tree = self.decision_trees[i]
            tree_predicts[:, i] = tree.predict(X)
        yhat = stats.mode(tree_predicts, axis=1).mode
        return yhat


class RandomForest(BaggedTrees):

    def __init__(self, n=200, m=1, params=None):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.m = m
        self.feature_idx = np.zeros((n,m), dtype=int)
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]
   
    def fit(self, X, y):
        # TODO implement function
        # pass
        for i in range(self.n):
            tree = self.decision_trees[i]
            rand_sample_idx = np.random.randint(low=0, high=len(y), size=len(y))
            # rand_feature_idx = np.random.choice(range(X.shape[1]), self.m, replace=False)
            rand_feature_idx = np.random.permutation(np.arange(X.shape[1]))[:self.m]
            self.feature_idx[i, :] = rand_feature_idx
            reduced_X = X[rand_sample_idx, :]
            reduced_X = reduced_X[:, rand_feature_idx]
            tree.fit(reduced_X, y[rand_sample_idx])

    def predict(self, X):
        tree_predicts = np.zeros((X.shape[0], self.n))
        yhat = np.zeros(X.shape[0])
        for i in range(self.n):
            tree = self.decision_trees[i]
            # print(self.feature_idx[i, :])
            reduced_X = X[:, self.feature_idx[i]]
            tree_predicts[:, i] = tree.predict(reduced_X)
        yhat = stats.mode(tree_predicts, axis=1).mode
        return yhat


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            # mm = stats.mode(data[(data[:, i] < -1 - eps) + (data[:, i] > -1 + eps)][:,i])
            # print(mm)
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode
            data[(data[:, i] > -1 - eps) *
                 (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf, num_splits=3):
    print("Cross validation", cross_val_score(clf, X, y, cv=num_splits))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def generate_submission(testing_data, predictions):
    # This code below will generate the predictions.csv file.
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('predictions.csv', index_label='Id')

    # Now download the predictions.csv file to submit.`

if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        "min_samples_leaf": 10,
        # "criterion": "entropy",
    }
    N = 200
    n_tries = 5
    validate_split = 0.2

    df = pd.DataFrame(
        columns=['Simple Tree', 'SKLearn Tree', 'Bagged Trees', 'Random Forest'],
        index=list(range(1, n_tries+1)))
    df.index.name = 'Run'
    df.style.set_caption(dataset)

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    from sklearn.model_selection import train_test_split

    for i in range(1, 1+n_tries):
        # split test data into train/validate
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=validate_split)

        print("\n\n----------------------------------------")
        # Basic decision tree
        print("\nPart (a-b): simplified decision tree")
        # TODO
        simple_tree = DecisionTree(max_depth=4, feature_labels=features)
        simple_tree.fit(X_train, y_train)

        simple_tree_yhat = simple_tree.predict(X_validate)
        simple_tree_accuracy = (y_validate == simple_tree_yhat).sum() / (1. * len(y_validate))
        df.at[i, 'Simple Tree'] = simple_tree_accuracy * 100
        print(f'Simple Decision Tree Accuracy: {simple_tree_accuracy}')

        # Basic decision tree
        print("\n\nPart (c): sklearn's decision tree")
        # Hint: Take a look at the imports!
        # clf = None # TODO
        clf = DecisionTreeClassifier(**params)
        clf.fit(X_train, y_train)
        clf_yhat = clf.predict(X_validate)
        clf_accuracy = (y_validate == clf_yhat).sum() / (1. * len(y_validate))
        df.at[i, 'SKLearn Tree'] = clf_accuracy * 100
        print(f'CLF Tree Accuracy: {clf_accuracy}')

        # TODO
        # Visualizing the tree
        out = io.StringIO()
        export_graphviz(
            clf, out_file=out, feature_names=features, class_names=class_names, filled=True)
        # For OSX, may need the following for dot: brew install gprof2dot
        graph = graph_from_dot_data(out.getvalue())
        graph_from_dot_data(out.getvalue())[0].write_pdf("%s-basic-tree.pdf" % dataset)

        # Bagged trees
        print("\n\nPart (d-e): bagged trees")
        # TODO
        bagged_trees = BaggedTrees(n=N, params=params)
        bagged_trees.fit(X_train, y_train)
        # print(bagged_trees)
        print(Counter([features[tt.tree_.feature[0]] for tt in bagged_trees.decision_trees]))
        bagged_trees_yhat = bagged_trees.predict(X_validate)
        # print(bagged_trees_yhat)
        bagged_trees_accuracy = (y_validate == bagged_trees_yhat).sum() / (1. * len(y_validate))
        df.at[i, 'Bagged Trees'] = bagged_trees_accuracy * 100
        print(f'Bagged Trees Accuracy: {bagged_trees_accuracy}')


        # Random forest
        print("\n\nPart (f-g): random forest")
        # TODO
        half_feature_size = int(X_train.shape[1] * 0.5)
        print(f"Using feature size: {half_feature_size}")
        random_forest = RandomForest(params=params, m=half_feature_size, n=N)
        random_forest.fit(X_train, y_train)
        print(Counter([features[tt.tree_.feature[0]] for tt in random_forest.decision_trees]))
        random_forest_yhat = random_forest.predict(X_validate)
        random_forest_accuracy = (y_validate == random_forest_yhat).sum() / (1. * len(y_validate))
        df.at[i, 'Random Forest'] = random_forest_accuracy * 100
        print(f'Random Forest Accuracy: {random_forest_accuracy}')
    
    print("\n\n----------------------------------------")
    df.loc['Avg'] = df.mean(axis=0)
    print(df)

    # Generate csv file of predictions on test data
    # TODO

