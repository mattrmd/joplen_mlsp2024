# Implemented by ChatGPT4

from typing import Callable, Tuple, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor


class RegressionTreeNode:
    def __init__(self):
        self.threshold = None
        self.feature_index = None
        self.left = None
        self.right = None
        self.output = None
        self.index = None  # New attribute for node index


class RegressionTree:
    def __init__(
        self,
        impurity_function: Optional[Callable] = None,
        min_samples_leaf: int = 2,
        max_depth=10,
    ):
        self.root = RegressionTreeNode()
        self.impurity_function = (
            impurity_function if impurity_function else self.default_impurity_function
        )
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def default_impurity_function(self, y: np.ndarray) -> float:
        """Default impurity function (Variance)"""
        return np.var(y)

    def best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[float], Optional[int]]:
        """Find the best split for a node"""
        min_impurity = np.inf
        best_threshold = None
        best_feature = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                # Skip the split if it does not meet the minimum samples requirement
                if (
                    len(y[left_mask]) < self.min_samples_leaf
                    or len(y[right_mask]) < self.min_samples_leaf
                ):
                    continue

                left_impurity = self.impurity_function(y[left_mask])
                right_impurity = self.impurity_function(y[right_mask])
                impurity = left_impurity * len(y[left_mask]) + right_impurity * len(
                    y[right_mask]
                )
                if impurity < min_impurity:
                    min_impurity = impurity
                    best_threshold = threshold
                    best_feature = feature_index

        return best_threshold, best_feature

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node: Optional[RegressionTreeNode] = None,
        depth: int = 0,
        index: int = 0,
    ):
        """Recursive function to build the tree"""
        if node is None:
            node = self.root

        if depth is None:
            depth = 0

        node.index = index  # Assign index to the node

        if depth == self.max_depth or len(y) == 0:
            node.output = np.mean(y)
            return

        threshold, feature = self.best_split(X, y)
        if threshold is None:
            node.output = np.mean(y)
            return

        node.threshold = threshold
        node.feature_index = feature
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        # Handle cases where a split results in an empty subset
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            node.output = np.mean(y)
            return

        # Assigning indices and recursively fitting left and right child nodes
        node.left = RegressionTreeNode()
        self.fit(X[left_mask], y[left_mask], node.left, depth + 1, index)

        node.right = RegressionTreeNode()
        self.fit(
            X[right_mask],
            y[right_mask],
            node.right,
            depth + 1,
            index + 2 ** (self.max_depth - (depth + 1)),
        )

    def predict(
        self, X: np.ndarray, node: Optional[RegressionTreeNode] = None
    ) -> np.ndarray:
        """Recursive function to make predictions"""
        if node is None:
            node = self.root

        if node.output is not None:
            return np.full(X.shape[0], node.output)

        mask = X[:, node.feature_index] <= node.threshold
        predictions = np.empty(X.shape[0])
        predictions[mask] = self.predict(X[mask], node.left)
        predictions[~mask] = self.predict(X[~mask], node.right)
        return predictions

    def predict_index(
        self, X: np.ndarray, node: Optional[RegressionTreeNode] = None
    ) -> np.ndarray:
        """Recursive function to predict the index of the node where each data point ends"""
        if node is None:
            node = self.root

        if node.output is not None:  # Leaf node
            return np.full(X.shape[0], node.index)

        mask = X[:, node.feature_index] <= node.threshold
        predictions = np.empty(X.shape[0], dtype=int)
        predictions[mask] = self.predict_index(X[mask], node.left)
        predictions[~mask] = self.predict_index(X[~mask], node.right)
        return predictions


class VarMaxForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        n_features: int | None = None,
        bootstrap_fraction: float = 1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.bootstrap_fraction = bootstrap_fraction
        self.random_state = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )

        self.trees = [
            RegressionTree(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                impurity_function=lambda x: -np.var(x) / x.shape[0],
            )
            for _ in range(n_estimators)
        ]

    def fit(self, X, y):
        if self.n_features is None:
            self.n_features = int(np.ceil(np.sqrt(X.shape[1])))

        feat_masks = []

        for tree in self.trees:
            feat_mask = self.random_state.choice(
                X.shape[1],
                size=self.n_features,
                replace=False,
            )
            sample_mask = self.random_state.choice(
                X.shape[0],
                size=int(self.bootstrap_fraction * X.shape[0]),
                replace=True,
            )
            tree.fit(X[sample_mask][:, feat_mask], y[sample_mask])
            feat_masks.append(feat_mask)

        self.feat_masks = np.array(feat_masks)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])

        for tree, mask in zip(self.trees, self.feat_masks):
            predictions += tree.predict(X[:, mask])
        return predictions / len(self.trees)

    def predict_index(self, X):
        predictions = np.zeros((len(self.trees), X.shape[0]))

        for i, (tree, mask) in enumerate(zip(self.trees, self.feat_masks)):
            predictions[i, :] = tree.predict_index(X[:, mask])

        return predictions


class LinearTreeNode:
    def __init__(self):
        self.threshold = None
        self.feature_index = None
        self.left = None
        self.right = None
        self.model = None  # Linear regression model
        self.index = None  # New attribute for node index


class LinearTree:
    def __init__(
        self,
        min_samples_leaf: int = 2,
        max_depth: int = 10,
    ):
        self.root = LinearTreeNode()
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def linear_model_impurity(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the impurity based on linear regression model error"""
        if len(y) < 2:  # Not enough data to fit a linear model
            return np.inf
        model = LinearRegression().fit(X, y)
        predictions = model.predict(X)
        return np.mean((y - predictions) ** 2)

    def best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[float], Optional[int], Optional[LinearRegression]]:
        """Find the best split for a node"""
        min_impurity = np.inf
        best_threshold = None
        best_feature = None
        best_model = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                if (
                    len(y[left_mask]) < self.min_samples_leaf
                    or len(y[right_mask]) < self.min_samples_leaf
                ):
                    continue

                model = LinearRegression()
                model.fit(X[left_mask], y[left_mask])
                left_impurity = self.linear_model_impurity(X[left_mask], y[left_mask])

                model.fit(X[right_mask], y[right_mask])
                right_impurity = self.linear_model_impurity(
                    X[right_mask], y[right_mask]
                )

                impurity = left_impurity * len(y[left_mask]) + right_impurity * len(
                    y[right_mask]
                )
                if impurity < min_impurity:
                    min_impurity = impurity
                    best_threshold = threshold
                    best_feature = feature_index
                    best_model = model

        return best_threshold, best_feature, best_model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node: Optional[LinearTreeNode] = None,
        depth: int = 0,
        index: int = 0,
    ):
        """Recursive function to build the tree"""
        if node is None:
            node = self.root

        if depth is None:
            depth = 0

        node.index = index  # Assign index to the node

        # Stopping condition for recursion
        if depth == self.max_depth or len(y) < self.min_samples_leaf:
            if len(y) >= 2:  # Enough data to fit a linear model
                node.model = LinearRegression().fit(X, y)
            return

        # Finding the best split
        threshold, feature, model = self.best_split(X, y)
        if threshold is None:  # No further splits possible
            if len(y) >= 2:  # Enough data to fit a linear model
                node.model = LinearRegression().fit(X, y)
            return

        node.threshold = threshold
        node.feature_index = feature
        node.model = model  # Store the linear regression model in the node

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        # Create left and right children and recursively fit them
        if np.any(left_mask):
            node.left = LinearTreeNode()
            self.fit(X[left_mask], y[left_mask], node.left, depth + 1, index * 2 + 1)

        if np.any(right_mask):
            node.right = LinearTreeNode()
            self.fit(X[right_mask], y[right_mask], node.right, depth + 1, index * 2 + 2)

        node.model = None

    def predict(
        self, X: np.ndarray, node: Optional[LinearTreeNode] = None
    ) -> np.ndarray:
        """Recursive function to make predictions"""
        if node is None:
            node = self.root

        if node.model is not None:  # Node has a linear model
            return node.model.predict(X)

        mask = X[:, node.feature_index] <= node.threshold
        predictions = np.empty(X.shape[0])

        # Predict separately for data falling in left and right child nodes
        if node.left is not None:
            predictions[mask] = self.predict(X[mask], node.left)
        else:  # Use parent node model if child node is missing
            predictions[mask] = node.model.predict(X[mask])

        if node.right is not None:
            predictions[~mask] = self.predict(X[~mask], node.right)
        else:  # Use parent node model if child node is missing
            predictions[~mask] = node.model.predict(X[~mask])

        return predictions

    def predict_index(
        self, X: np.ndarray, node: Optional[LinearTreeNode] = None
    ) -> np.ndarray:
        """Recursive function to predict the index of the node where each data point ends"""
        if node is None:
            node = self.root

        if node.model is not None:  # Leaf node
            return np.full(X.shape[0], node.index)

        mask = X[:, node.feature_index] <= node.threshold
        predictions = np.empty(X.shape[0], dtype=int)
        predictions[mask] = self.predict_index(X[mask], node.left)
        predictions[~mask] = self.predict_index(X[~mask], node.right)
        return predictions


class LinearForestRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        n_features: Optional[int] = None,
        bootstrap_fraction: float = 1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.bootstrap_fraction = bootstrap_fraction
        self.random_state = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        self.feat_masks = []

    def fit_tree(self, X, y, feat_mask, sample_mask):
        tree = LinearTree(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
        )
        tree.fit(X[sample_mask][:, feat_mask], y[sample_mask])
        return tree, feat_mask

    def fit(self, X, y):
        if self.n_features is None:
            self.n_features = int(np.ceil(np.sqrt(X.shape[1])))

        with ProcessPoolExecutor() as executor:
            futures = []
            for _ in range(self.n_estimators):
                feat_mask = self.random_state.choice(
                    X.shape[1],
                    size=self.n_features,
                    replace=False,
                )
                sample_mask = self.random_state.choice(
                    X.shape[0],
                    size=int(self.bootstrap_fraction * X.shape[0]),
                    replace=True,
                )
                futures.append(
                    executor.submit(self.fit_tree, X, y, feat_mask, sample_mask)
                )

            results = [future.result() for future in futures]
            self.trees, self.feat_masks = zip(*results)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree, mask in zip(self.trees, self.feat_masks):
            predictions += tree.predict(X[:, mask])
        return predictions / self.n_estimators

    def predict_index(self, X):
        predictions = np.zeros((len(self.trees), X.shape[0]))

        for i, (tree, mask) in enumerate(zip(self.trees, self.feat_masks)):
            predictions[i, :] = tree.predict_index(X[:, mask])

        return predictions


class GBLTRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        base_estimator=LinearRegression(),
    ):
        # Initialize the base learners as Linear Tree Regressors
        self.base_learners = [
            LinearTreeRegressor(
                max_depth=max_depth,
                base_estimator=base_estimator,
            )
            for _ in range(n_estimators)
        ]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # Initial residuals are just the original labels
        residuals = y

        for learner in self.base_learners:
            # Fit learner to the residuals
            learner.fit(X, residuals)

            # Update residuals
            predictions = learner.predict(X)
            residuals = residuals - self.learning_rate * predictions

    def predict(self, X):
        # Start with all zeros
        predictions = np.zeros(X.shape[0])

        for learner in self.base_learners:
            # Add predictions from each learner
            predictions += self.learning_rate * learner.predict(X)

        return predictions
