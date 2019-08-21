import random
import numpy
from matplotlib import pyplot
from pandas import DataFrame


SEED = 1337


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
    pyplot.show()


def main():
    data, target = generate_random_data(20, 20, 1000)
    visualise_data(data, target)


if __name__ == '__main__':
    main()
