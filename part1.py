import pandas
from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import Bunch


SEED = 1337


def split_data(dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.6, random_state=SEED)
    training_data = Bunch()
    training_data.data = X_train
    training_data.target = y_train
    if 'feature_names' in dataset:
        training_data.feature_names = dataset.feature_names
    training_data.target_names = dataset.target_names

    testing_data = Bunch()
    testing_data.data = X_test
    testing_data.target = y_test
    if 'feature_names' in dataset:
        testing_data.feature_names = dataset.feature_names
    testing_data.target_names = dataset.target_names

    return training_data, testing_data


def evaluate_depth(training_dataset, max_depth, k):
    X, y = training_dataset.data, training_dataset.target
    num_nodes = []
    accuracies = []

    kfold = KFold(n_splits=k, random_state=SEED)
    for train_index, test_index in kfold.split(X, y):
        # Get training/testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Construct a decision tree model
        decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=SEED).fit(X_train, y_train)

        # Find num of nodes / accuracy
        num_nodes.append(decision_tree.tree_.node_count)
        y_predictions = decision_tree.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_predictions)
        accuracies.append(accuracy)

    avg_num_nodes = sum(num_nodes) / k
    avg_accuracy = sum(accuracies) / k
    return avg_accuracy, avg_num_nodes


def display_tree(decision_tree, dataset):
    feature_names = dataset.get('feature_names', list(f'X{i}' for i in range(dataset.data.shape[1])))
    target_names = [str(name) for name in dataset.target_names]
    plot_tree(decision_tree, feature_names=feature_names, class_names=target_names)


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

    print(pandas.DataFrame(output).to_string(index=False, float_format='{:.4f}'.format))


def find_best_max_depth(results):
    curr_accuracy = 0

    for max_depth, result in results.items():
        accuracy, num_nodes = result
        if accuracy - curr_accuracy > 0.01:
            curr_accuracy = accuracy
        else:
            return max_depth, curr_accuracy


def main():
    sets = {
        'Iris': datasets.load_iris(),
        'Breast Cancer': datasets.load_breast_cancer(),
        'Digits': datasets.load_digits()
    }

    for name, dataset in sets.items():
        training_partition, testing_partition = split_data(dataset)

        results = dict()
        for max_depth in range(1, 20 + 1):
            avg_accuracy, avg_num_nodes = evaluate_depth(training_partition, max_depth=max_depth, k=7)
            results[max_depth] = (avg_accuracy, avg_num_nodes)

        print(name)
        print_results(results)
        best_max_depth, best_accuracy = find_best_max_depth(results)
        print(f'Best max depth: {best_max_depth}')

        # Using best max depth, train a decision tree using the entire dataset
        decision_tree = DecisionTreeClassifier(max_depth=best_max_depth, random_state=SEED)\
            .fit(training_partition.data, training_partition.target)
        y_predictions = decision_tree.predict(testing_partition.data)
        overall_accuracy = metrics.accuracy_score(testing_partition.target, y_predictions)
        print(f'Overall accuracy of classifier: {overall_accuracy:.4f}')
        print(f'Number of nodes in overall tree: {decision_tree.tree_.node_count}')


if __name__ == '__main__':
    main()
