# Gradient Boosted Trees Implementation

## Project Overview

This project implements gradient boosted trees for binary classification from first principles, following the algorithm described in Sections 10.9-10.10 of "Elements of Statistical Learning" (2nd Edition). The implementation provides a scikit-learn-like interface with fit and predict methods, making it easy to use for classification tasks.

## Project Structure

```
spring2025cs584-project2/
├── README.md
├── requirements.txt
├── GradientBoost/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── GradientBoostedTrees.py
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_GradientBoostedTrees.py
│   ├── datasets/
│   │   └── generate_datasets.py
│   └── notebooks/
│       └── GBT_visualization.ipynb
```

## Running the Code

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the tests:

   ```
   python -m GradientBoost.tests.test_GradientBoostedTrees
   ```

3. Generate datasets and visualizations:

   ```
   python -m GradientBoost.datasets.generate_datasets
   ```

4. Explore the Jupyter notebook for additional visualizations:
   ```
   jupyter notebook GradientBoost/notebooks/GBT_visualization.ipynb
   ```

## Basic Usage

```python
# Import the model
from GradientBoost.model.GradientBoostedTrees import GradientBoostedTrees
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# Create a simple dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model with default parameters
gbt = GradientBoostedTrees(n_estimators=100, learning_rate=0.1, max_depth=3)
gbt.fit(X_train, y_train)

# Make predictions
y_pred = gbt.predict(X_test)
probabilities = gbt.predict_proba(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Visualization Results

The notebook in the `notebooks` directory provides extensive visualizations of the model's performance. Here are some key results:

**Simple Dataset**

- Final Accuracy: 0.9250
- AUC: 0.9794
- Training Time: ~45.5s

**Moons Dataset**

- Final Accuracy: 0.9350
- AUC: 0.9758
- Training Time: ~44.9s

**Cancer Dataset**

- Final Accuracy: 0.9561
- AUC: 0.993
- Training Time: ~339.5s

## Questions and Answers

### What does the model you have implemented do and when should it be used?

Gradient Boosted Trees is an ensemble learning technique that builds multiple decision trees sequentially, where each tree tries to correct the errors made by the previous trees. The key characteristics of this implementation include:

- **Binary classification**: The model is specifically designed for binary classification problems, mapping class labels to -1 and 1 internally.
- **Sequential learning**: Trees are built in sequence, with each new tree correcting errors of the previously built ensemble.
- **Gradient descent in function space**: The model uses gradient descent to minimize the loss function by adding trees that predict the negative gradient (pseudo-residuals).
- **Weight-based learning**: Sample weights are calculated based on the absolute values of gradients to focus on high-error samples.

This model should be used when:

1. You have a binary classification problem
2. Your data may have complex, non-linear relationships
3. You need a model that can handle diverse types of features without extensive preprocessing
4. You want good out-of-the-box performance with the ability to tune parameters for better results

Gradient boosted trees typically perform well on structured tabular data, handling both linear and non-linear relationships. They're particularly effective when there are complex interactions between features that simpler models might miss.

### How did you test your model to determine if it is working reasonably correctly?

The model was tested using several approaches:

1. **Simple classification datasets**: Testing on basic classification datasets created with `make_classification()`.
2. **Complex non-linear datasets**: Testing on datasets with non-linear decision boundaries (`make_moons()`, `make_circles()`).
3. **Real-world dataset**: Testing on the Breast Cancer Wisconsin dataset.
4. **Collinear data**: Testing on datasets with highly correlated features to check how the model handles collinearity.
5. **Parameter sensitivity**: Testing different learning rates and tree depths to understand parameter impacts.
6. **Visualization**: Decision boundaries and ROC curves are plotted to visualize model performance.

The test results demonstrate:

- Consistent accuracy scores above 90% across different datasets
- Ability to capture non-linear decision boundaries
- Good handling of collinear features
- Expected behavior with parameter changes

Comprehensive test scripts are provided in the `tests` directory, and visualization notebooks are available in the `notebooks` directory.

### What parameters have you exposed to users of your implementation in order to tune performance?

The `GradientBoostedTrees` class exposes several parameters to tune the model's performance:

- **n_estimators** (default=100): The number of boosting stages (trees) to build. Increasing this parameter often improves performance but at the cost of longer training time and potentially overfitting.

- **learning_rate** (default=0.1): Shrinks the contribution of each tree. Lower values require more trees but can lead to better generalization. There's a trade-off between learning_rate and n_estimators.

- **max_depth** (default=3): Maximum depth of the individual regression trees. Deeper trees can capture more complex patterns but might lead to overfitting.

- **min_samples_split** (default=2): Minimum number of samples required to split an internal node. Higher values prevent creating too specific splits.

- **min_samples_leaf** (default=1): Minimum number of samples required to be at a leaf node. Higher values prevent creating leaves with very few samples.

- **min_impurity_decrease** (default=0.0): A node will be split only if the split decreases the impurity by at least this value.

- **random_state** (default=None): Controls the randomness of the estimator.

### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

#### Inputs that the implementation has trouble with:

1. **Multi-class classification**: The current implementation only handles binary classification. It would need significant modification to handle multi-class problems.

2. **Very large datasets**: The implementation doesn't include the optimization techniques that production-grade libraries use, so it may be slower on very large datasets.

3. **High-dimensional sparse data**: While it can handle such data, performance might degrade as the dimensionality increases.

4. **Categorical features**: The current implementation expects numerical features; categorical features would need to be encoded before use.

5. **Imbalanced datasets**: The implementation doesn't have specific strategies for handling class imbalance.

#### Potential improvements (with more time):

1. **Multi-class classification**: Implement one-vs-rest or multi-class gradient boosting approaches.

2. **Feature importance**: Add methods to calculate and visualize feature importance.

3. **Regularization techniques**: Implement additional regularization options like L1/L2 regularization on leaf weights.

4. **Early stopping**: Add early stopping based on validation metrics to prevent overfitting.

5. **Sub-sampling**: Implement stochastic gradient boosting with row and column sub-sampling.

6. **Optimized tree building**: Improve the efficiency of the tree-building algorithm for faster training.

7. **Cross-validation**: Add built-in cross-validation capabilities for parameter tuning.

8. **Handling missing values**: Extend the implementation to handle missing values directly.

Most of these limitations could be addressed with additional development time, though some (like performance on very large datasets) would require more substantial optimization.

---

### Submitted by

- **Vishwashree Channaa Reddy Hanumanthareddy** - A20556543
- **Ananth Krishna Vasireddy** - A20585441
- **Aishwarya Ainala** - A20546437
- **Yasaswini Kakumani** - A20547678
