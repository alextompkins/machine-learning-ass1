from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold
from matplotlib import pyplot


def evaluate_decision_tree(dataset, max_depth, k):
    X, y = dataset.data, dataset.target
    num_nodes = []
    accuracies = []

    kfold = KFold(n_splits=k)
    for train_index, test_index in kfold.split(X, y):
        # Get training/testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Construct a decision tree model
        decision_tree = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)

        # Find num of nodes / accuracy
        num_nodes.append(decision_tree.tree_.node_count)
        y_predictions = decision_tree.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_predictions)
        accuracies.append(accuracy)

        plot_tree(decision_tree, feature_names=dataset.feature_names, class_names=dataset.target_names)

    avg_num_nodes = sum(num_nodes) / k
    avg_accuracy = sum(accuracies) / k
    return avg_accuracy, avg_num_nodes


def main():
    iris = datasets.load_iris()

    for max_depth in range(1, 50):
        avg_accuracy, avg_num_nodes = evaluate_decision_tree(iris, max_depth=max_depth, k=7)

        print(f'Max depth: {max_depth}')
        print(f'Avg num nodes: {avg_num_nodes:.1f}')
        print(f"Avg accuracy: {avg_accuracy:.3f}")

    # pyplot.show()


if __name__ == '__main__':
    main()
