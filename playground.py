import pandas
from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import Bunch
from matplotlib import pyplot


def split_data(dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)
    training_data = Bunch()
    training_data.data = X_train
    training_data.target = y_train
    training_data.feature_names = dataset.feature_names
    training_data.target_names = dataset.target_names

    testing_data = Bunch()
    testing_data.data = X_test
    testing_data.target = y_test
    testing_data.feature_names = dataset.feature_names
    testing_data.target_names = dataset.target_names

    return training_data, testing_data


def evaluate_depth(training_dataset, max_depth, k):
    X, y = training_dataset.data, training_dataset.target
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

        plot_tree(decision_tree, feature_names=training_dataset.feature_names, class_names=training_dataset.target_names)

    avg_num_nodes = sum(num_nodes) / k
    avg_accuracy = sum(accuracies) / k
    return avg_accuracy, avg_num_nodes


def print_results(results):
    max_depth_label = 'Max Depth'
    accuracy_label = 'Accuracy'
    num_nodes_label = '# Nodes'

    output = {
        max_depth_label: [],
        accuracy_label: [],
        num_nodes_label: []
    }

    for max_depth in results.keys():
        output[max_depth_label].append(max_depth)
        output[accuracy_label].append(results[max_depth][0])
        output[num_nodes_label].append(results[max_depth][1])

    print(pandas.DataFrame(output).to_string(index=False, float_format='{:.3f}'.format))


# def find_best_max_depth(results):
#     for


def main():
    iris = datasets.load_iris()
    training_partition, testing_partition = split_data(iris)

    results = dict()
    for max_depth in range(1, 20 + 1):
        avg_accuracy, avg_num_nodes = evaluate_depth(training_partition, max_depth=max_depth, k=7)
        results[max_depth] = (avg_accuracy, avg_num_nodes)

    print_results(results)
    # pyplot.show()


if __name__ == '__main__':
    main()
