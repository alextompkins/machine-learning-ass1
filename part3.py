import numpy
import pandas
from sklearn import linear_model, metrics


FILENAME = 'assignment1-2019-data.csv'


def read_data_from_csv():
    dataframe = pandas.read_csv(FILENAME)
    return dataframe


def create_piecewise_function(c, intercept):
    """
    Creates the piecewise function identified by the regression model. This function will have different weights for
    each input variable based on the value of the categorical variable.
    :param c: The learned coefficients of each input variable
    :param intercept: The y-intercept of the learned model
    :return: The piecewise function learned by the regression model
    """
    weights = numpy.array([
        [c['X1'], c['X2'], c['X3'], intercept],
        [c['X1'] + c['X1B'], c['X2'] + c['X2B'], c['X3'] + c['X3B'], intercept + c['DummyB']],
        [c['X1'] + c['X1C'], c['X2'] + c['X2C'], c['X3'] + c['X3C'], intercept + c['DummyC']]
    ])

    def predict(row):
        weight_map = {
            'A': weights[0],
            'B': weights[1],
            'C': weights[2]
        }
        coeffs = weight_map[row['X4']]
        return coeffs[0] * row['X1'] + coeffs[1] * row['X2'] + coeffs[2] * row['X3'] + coeffs[3]

    print(weights)
    return predict


def main():
    dataframe = read_data_from_csv()

    # Add dummy variables and interactions terms based on the 3 levels of the categorical variable
    # These will also be learned by the model to provide greater accuracy
    dataframe['DummyB'] = dataframe['X4'].apply(lambda category: 1 if category == 'B' else 0)
    dataframe['DummyC'] = dataframe['X4'].apply(lambda category: 1 if category == 'C' else 0)
    for col in ('X1', 'X2', 'X3'):
        dataframe[f'{col}B'] = dataframe[col] * dataframe['DummyB']
        dataframe[f'{col}C'] = dataframe[col] * dataframe['DummyC']

    # Carry out linear regression to find coefficients and intercept
    x = dataframe[[col for col in dataframe if col not in ('X4', 'Y')]]
    y = dataframe['Y']
    regression = linear_model.LinearRegression().fit(x, y)
    coeffs = {x.columns.values[i]: int(round(coeff)) for i, coeff in enumerate(regression.coef_)}
    intercept = int(round(regression.intercept_))

    print(coeffs)
    print(f'Intercept: {intercept}')

    # Test accuracy of our learned model
    predict = create_piecewise_function(coeffs, intercept)
    predictions = dataframe[['X1', 'X2', 'X3', 'X4']].apply(predict, axis=1)
    mean_sq_error = metrics.mean_squared_error(y, predictions)
    r_squared_score = metrics.r2_score(y, predictions)
    print(f'Mean Squared Error: {mean_sq_error}\n'
          f'R^2 score: {r_squared_score}')


if __name__ == '__main__':
    main()
