import numpy
import pandas
from sklearn import linear_model


FILENAME = 'assignment1-2019-data.csv'


def read_data_from_csv():
    dataframe = pandas.read_csv(FILENAME)
    return dataframe


def main():
    dataframe = read_data_from_csv()
    dataframe['DummyB'] = dataframe['X4'].apply(lambda category: 1 if category == 'B' else 0)
    dataframe['DummyC'] = dataframe['X4'].apply(lambda category: 1 if category == 'C' else 0)

    for col in ('X1', 'X2', 'X3'):
        dataframe[f'{col}B'] = dataframe[col] * dataframe['DummyB']
        dataframe[f'{col}C'] = dataframe[col] * dataframe['DummyC']

    x = dataframe[[col for col in dataframe if col not in ('X4', 'Y')]]
    y = dataframe['Y']

    regression = linear_model.LinearRegression().fit(x, y)
    print(regression)
    print(x.columns.values)
    print(regression.intercept_)
    print(regression.coef_)

    w = {x.columns.values[i]: coeff for i, coeff in enumerate(regression.coef_)}

    weights = numpy.array([
        [w['X1'], w['X2'], w['X3'], regression.intercept_],
        [w['X1'] + w['X1B'], w['X2'] + w['X2B'], w['X3'] + w['X3B'], regression.intercept_ + w['DummyB']],
        [w['X1'] + w['X1C'], w['X2'] + w['X2C'], w['X3'] + w['X3C'], regression.intercept_ + w['DummyC']]
    ])

    def y(x):
        coeffs = [0, 0, 0, 0]
        if x[3] == 'A':
            coeffs = weights[0]
        elif x[3] == 'B':
            coeffs = weights[1]
        elif x[3] == 'C':
            coeffs = weights[2]
        return coeffs[0] * x[0] + coeffs[1] * x[1] + coeffs[2] * x[2] + coeffs[3] * regression.intercept_

    print(weights)

    # b1 intercept for A (base case)
    # b2 is difference in intercept between B and A
    # b3 is difference in intercept between C and A


if __name__ == '__main__':
    main()
