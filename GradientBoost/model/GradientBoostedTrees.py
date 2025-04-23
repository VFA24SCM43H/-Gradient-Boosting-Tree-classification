import numpy as np
from collections import Counter
import time

class TreeNode:
    """
    Node class for decision tree
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Value to return if this is a leaf node

    def is_leaf(self):
        """Check if the node is a leaf node"""
        return self.value is not None

class DecisionTree:
    """
    Decision Tree implementation for gradient boosting
    """
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1, 
                 min_impurity_decrease=0.0, criterion='mse'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.root = None
        
    def fit(self, X, y, sample_weight=None):
        """
        Build the decision tree
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (pseudo-residuals in gradient boosting).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns:
        --------
        self : object
        """
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, sample_weight, depth=0)
        return self
    
    def _build_tree(self, X, y, sample_weight=None, depth=0):
        """
        Recursively build the decision tree
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (pseudo-residuals).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        depth : int
            Current depth of the tree.
            
        Returns:
        --------
        node : TreeNode
            The root node of the built tree.
        """
        n_samples, n_features = X.shape
        
        # Use uniform weights if none provided
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
            
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.all(y == y[0])):
            return TreeNode(value=self._calculate_leaf_value(y, sample_weight))
        
        # Find the best split
        feature_idx, threshold, impurity_decrease = self._find_best_split(X, y, sample_weight)
        
        # If no good split is found, create a leaf node
        if feature_idx is None or impurity_decrease < self.min_impurity_decrease:
            return TreeNode(value=self._calculate_leaf_value(y, sample_weight))
        
        # Split the data
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices
        
        # Check if split results in valid leaf nodes
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            return TreeNode(value=self._calculate_leaf_value(y, sample_weight))
        
        # Recursively build the left and right subtrees
        left = self._build_tree(
            X[left_indices], 
            y[left_indices], 
            sample_weight[left_indices] if sample_weight is not None else None,
            depth + 1
        )
        right = self._build_tree(
            X[right_indices], 
            y[right_indices], 
            sample_weight[right_indices] if sample_weight is not None else None,
            depth + 1
        )
        
        return TreeNode(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
    
    def _find_best_split(self, X, y, sample_weight):
        """
        Find the best split for a node
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (pseudo-residuals).
        sample_weight : array-like of shape (n_samples,)
            Sample weights.
            
        Returns:
        --------
        best_feature : int or None
            The index of the best feature to split on.
        best_threshold : float or None
            The threshold value for the best split.
        best_impurity_decrease : float
            The decrease in impurity for the best split.
        """
        n_samples, n_features = X.shape
        
        # Initialize variables to track the best split
        best_feature = None
        best_threshold = None
        best_impurity_decrease = -np.inf
        
        # Calculate the weighted impurity of the node before splitting
        parent_impurity = self._calculate_impurity(y, sample_weight)
        
        for feature_idx in range(n_features):
            # Sort the feature values and corresponding targets
            sorted_idx = np.argsort(X[:, feature_idx])
            sorted_X = X[sorted_idx, feature_idx]
            sorted_y = y[sorted_idx]
            sorted_weight = sample_weight[sorted_idx]
            
            # Skip if all values are the same
            if sorted_X[0] == sorted_X[-1]:
                continue
            
            # Find potential split points (midpoints between different values)
            unique_values = np.unique(sorted_X)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                # Split the data
                left_mask = sorted_X <= threshold
                right_mask = ~left_mask
                
                # Skip if split would create leaves with too few samples
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity for each child
                left_impurity = self._calculate_impurity(sorted_y[left_mask], sorted_weight[left_mask])
                right_impurity = self._calculate_impurity(sorted_y[right_mask], sorted_weight[right_mask])
                
                # Calculate weighted average of child impurities
                n_left = np.sum(sorted_weight[left_mask])
                n_right = np.sum(sorted_weight[right_mask])
                n_total = n_left + n_right
                
                # Calculate impurity decrease
                impurity_decrease = parent_impurity - (n_left / n_total * left_impurity + 
                                                    n_right / n_total * right_impurity)
                
                # Update the best split if this one is better
                if impurity_decrease > best_impurity_decrease:
                    best_impurity_decrease = impurity_decrease
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_impurity_decrease
    
    def _calculate_impurity(self, y, sample_weight):
        """
        Calculate the impurity of a node
        
        Parameters:
        -----------
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,)
            Sample weights.
            
        Returns:
        --------
        impurity : float
            The impurity value.
        """
        if self.criterion == 'mse':
            # Mean squared error (for regression)
            weighted_mean = np.average(y, weights=sample_weight)
            return np.average((y - weighted_mean) ** 2, weights=sample_weight)
        else:
            raise ValueError(f"Criterion '{self.criterion}' not supported")
    
    def _calculate_leaf_value(self, y, sample_weight):
        """
        Calculate the value for a leaf node
        
        Parameters:
        -----------
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,)
            Sample weights.
            
        Returns:
        --------
        value : float
            The leaf node value.
        """
        # For regression trees (used in gradient boosting), the leaf value is 
        # the weighted average of the target values
        return np.average(y, weights=sample_weight)
    
    def predict(self, X):
        """
        Predict using the decision tree
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def _predict_single(self, x, node):
        """
        Predict for a single sample using the decision tree
        
        Parameters:
        -----------
        x : array-like of shape (n_features,)
            The input sample.
        node : TreeNode
            The current node.
            
        Returns:
        --------
        value : float
            The predicted value.
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


class GradientBoostedTrees:
    """
    Gradient Boosted Trees implementation from first principles.
    
    This implementation follows the algorithm described in Sections 10.9-10.10
    of "Elements of Statistical Learning" (2nd Edition). It implements gradient
    boosting for binary classification using decision trees as base learners.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        The number of boosting stages to perform (number of trees).
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree. There is a trade-off
        between learning_rate and n_estimators.
    max_depth : int, default=3
        Maximum depth of the individual regression trees.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    random_state : int, default=None
        Controls the randomness of the estimator.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.trees = []
        self.initial_value = None
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Build a gradient boosted model from the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification).
            
        Returns:
        --------
        self : object
        """
        start_time = time.time()
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Convert y to {-1, 1} for binary classification
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostedTrees is currently only implemented for binary classification")
        
        # Map classes to -1 and 1
        y_binary = np.ones(len(y))
        y_binary[y == self.classes_[0]] = -1
        
        # Initialize the model with the log odds of the positive class
        pos_count = np.sum(y_binary == 1)
        neg_count = np.sum(y_binary == -1)
        self.initial_value = 0.5 * np.log(pos_count / neg_count)
        
        # Initialize predictions with the initial value
        F = np.full(len(y_binary), self.initial_value)
        
        print(f"Training gradient boosted trees with {self.n_estimators} estimators...")
        
        # Boosting iterations
        for m in range(self.n_estimators):
            # Compute negative gradients (pseudo-residuals)
            # For binary classification with log loss: gradient = y / (1 + exp(y * F))
            negative_gradients = y_binary / (1 + np.exp(y_binary * F))
            
            # Fit a regression tree to the pseudo-residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                criterion='mse'
            )
            
            # Calculate sample weights based on the absolute value of gradients
            sample_weight = np.abs(negative_gradients)
            
            # Fit the tree to the negative gradients
            tree.fit(X, negative_gradients, sample_weight=sample_weight)
            
            # Get predictions from the tree
            tree_predictions = tree.predict(X)
            
            # Line search to find the optimal step size (gamma) for each terminal region
            # For simplicity, we'll use a fixed learning rate instead of a full line search
            F += self.learning_rate * tree_predictions
            
            # Store the tree
            self.trees.append(tree)
            
            if (m + 1) % 10 == 0 or m == 0 or m == self.n_estimators - 1:
                # Calculate current predictions and error
                y_pred = self.predict(X)
                accuracy = np.mean(y_pred == y)
                print(f"Iteration {m+1}/{self.n_estimators}, Accuracy: {accuracy:.4f}")
        
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        X = np.asarray(X)
        
        # Start with the initial value
        F = np.full(X.shape[0], self.initial_value)
        
        # Add predictions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        # Convert log-odds to probabilities using the sigmoid function
        proba_pos = 1 / (1 + np.exp(-2 * F))
        
        # Return probabilities for both classes
        return np.column_stack([1 - proba_pos, proba_pos])
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]