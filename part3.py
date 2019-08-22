import numpy
import pandas
from sklearn import linear_model


FILENAME = 'assignment1-2019-data.csv'


def read_data_from_csv():
    dataframe = pandas.read_csv(FILENAME)
    return dataframe


def main():
    dataframe = read_data_from_csv()
    dataframe['dummy1'] = dataframe['X4'].apply(lambda category: 1 if category == 'B' else 0)
    dataframe['dummy2'] = dataframe['X4'].apply(lambda category: 1 if category == 'C' else 0)

    x = dataframe[['X1', 'X2', 'X3', 'dummy1', 'dummy2']]
    y = dataframe['Y']

    regression = linear_model.LinearRegression().fit(x, y)
    print(regression)
    print(regression.intercept_)
    print(regression.coef_)


if __name__ == '__main__':
    main()
