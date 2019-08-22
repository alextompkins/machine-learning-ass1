import random
import numpy
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from helpers import draw_line
from plot_decision_tree import decision_areas, plot_areas


SEED = 1337
TRAINING_SIZE_LABEL = 'Training Size'
MAX_DEPTH_LABEL = 'Max Depth'
ACCURACY_LABEL = 'Accuracy'


def classifier(x):
    """x - y + 4 < 0"""
    return x[0] - x[1] + 4 < 0


def generate_random_data(max_x1, max_x2, num_points):
    random.seed(SEED)
    data = numpy.asarray([(random.random() * max_x1, random.random() * max_x2) for i in range(num_points)])
    target = numpy.asarray([classifier(pt) for pt in data])
    return data, target


def visualise_data(data, target):
    colours = {
        True: 'green',
        False: 'red'
    }
    data_frame = DataFrame(dict(x=data[:, 0], y=data[:, 1], label=target))
    fig, ax = pyplot.subplots()
    grouped = data_frame.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colours[key])


def main():
    data, target = generate_random_data(20, 20, 2000)

    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.6, random_state=SEED)
    results = []
    tree_to_display = None

    for training_size in (5, 10, 50, 100, 500, 1000):
        X_train_resized = X_train[:training_size]
        y_train_resized = y_train[:training_size]

        for max_depth in range(1, 10 + 1):
            decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=SEED)\
                .fit(X_train_resized, y_train_resized)
            y_predictions = decision_tree.predict(X_test)
            accuracy = accuracy_score(y_test, y_predictions)
            results.append({
                TRAINING_SIZE_LABEL: training_size,
                MAX_DEPTH_LABEL: max_depth,
                ACCURACY_LABEL: accuracy
            })

            if max_depth == 10 and training_size == 1000:
                tree_to_display = decision_tree
                visualise_data(X_train_resized, y_train_resized)

    # Print pivoted table of accuracy for each Max Depth / Training Size combo
    data_frame = DataFrame(results)
    pivoted = data_frame.pivot_table(index=MAX_DEPTH_LABEL, columns=TRAINING_SIZE_LABEL)
    print(pivoted.to_string(float_format='{:.4f}'.format))

    # Add rectangles to plot to show decision tree boundaries
    rectangles = decision_areas(tree_to_display, (0, 20, 0, 20))
    plot_areas(rectangles)

    # Add line to chart to show classifier
    draw_line((0, 4), (16, 20), color='yellow', linewidth=2)

    # Show plot
    pyplot.xlim(0, 20)
    pyplot.ylim(0, 20)
    pyplot.show()


if __name__ == '__main__':
    main()
