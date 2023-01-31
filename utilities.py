import itertools
import numpy as np


def create_test_matrix_11(rows, cols, p=0.5):
    """
    Returns a Boolean ({+1, -1}) random matrix of size rows * columns
    :param rows: number of rows
    :param cols: number of columns
    :param p: probability
    :return: a matrix with random bits of size rows * columns
    """
    return 1 - 2 * np.random.binomial(1, p, size=(rows, cols)).astype(np.float32)


def create_test_matrix_cond(rows, cols, cond, p=0.5):
    """
    Creates a random matrix satisfying a certain condition.
    Condition should be something like X[:, 0] == 1
    """
    X = create_test_matrix_11(rows, cols, p)
    X = X[cond(X)]
    while len(X) < rows:
        temp = create_test_matrix_11(rows // 2, cols, p)
        temp = temp[cond(temp)]
        X = np.vstack([X, temp])
    return X[: rows, :]



def generate_all_binaries(d):
    """
    Generate all binary sequences of length d where bits are +1 and -1
    :param d: dimension
    :return: the output is a numpy array of size 2^d * d
    """
    return np.array([list(seq) for seq in itertools.product([-1, 1], repeat=d)], dtype=np.float32)



def calculate_fourier_coefficients(monomials, X, y):
    """
    calculate Fourier coefficients of monomials
    :param monomials: m * d, a mask to show which monomials we want. m is the number of monomials
    :param X: input data
    :param y: output data, y=f(X)
    :return: Fourier coefficient of the monomials which were indicated by monomials in the arguments.
    """
    return ((-2 * ((monomials @ ((1 - X.T) / 2)) % 2) + 1) @ y) / y.shape[0]



