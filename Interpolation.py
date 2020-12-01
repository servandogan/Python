import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    for i in range(x.size):
        polynom = np.poly1d([1])
        for j in range(x.size):
            if i != j:
                polynom = polynom * np.poly1d([1, -x[j]]) / (x[i] - x[j])

        base_functions.append(polynom)
        polynomial += polynom * y[i]

    # TODO: Generate Lagrange base polynomials and interpolation polynomial


    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []

    for i in range(x.size-1):
        A = np.zeros((4,4))

        A[0][:] = x[i] ** 3, x[i] ** 2, x[i], 1
        A[1][:] = x[i + 1] ** 3, x[i + 1] ** 2, x[i + 1], 1
        A[2][:] = 3 * x[i] ** 2, 2 * x[i], 1, 0
        A[3][:] = 3 * x[i + 1] ** 2, 2 * x[i + 1], 1, 0


        c = np.array([y[i], y[i + 1], yp[i], yp[i + 1]])

        M = np.linalg.solve(A, c)

        spline.append(np.poly1d(M))
    return spline




def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    g = x.size-1
    A = np.zeros((4 * g, 4 * g), dtype = np.float64)

    for i in range(g):

        j = i + 1
        A[4 * i][4 * i + 0] = 1
        A[4 * i][4 * i + 1] = x[i]
        A[4 * i][4 * i + 2] = x[i]**2
        A[4 * i][4 * i + 3] = x[i]**3

        A[4 * i + 1][4 * i + 0] = 1
        A[4 * i + 1][4 * i + 1] = x[j]
        A[4 * i + 1][4 * i + 2] = x[j] ** 2
        A[4 * i + 1][4 * i + 3] = x[j] ** 3
####################
        if i < (g - 1):
            A[4 * i + 2][4 * i] = 0
            A[4 * i + 2][4 * i + 1] = 1
            A[4 * i + 2][4 * i + 2] = 2 * x[j]
            A[4 * i + 2][4 * i + 3] = 3 * x[j]**2

            A[4 * i + 2][4 * i + 4] = 0
            A[4 * i + 2][4 * i + 5] = -1
            A[4 * i + 2][4 * i + 6] = -2 * x[j]
            A[4 * i + 2][4 * i + 7] = -3 * x[j] ** 2
####################
            A[4 * i + 3][4 * i] = 0
            A[4 * i + 3][4 * i + 1] = 0
            A[4 * i + 3][4 * i + 2] = 2
            A[4 * i + 3][4 * i + 3] = 6 * x[j]

            A[4 * i + 3][4 * i + 4] = 0
            A[4 * i + 3][4 * i + 5] = 0
            A[4 * i + 3][4 * i + 6] = -2
            A[4 * i + 3][4 * i + 7] = -6 * x[j]

    A[4 * x.size - 6][2] = 2
    A[4 * x.size - 6][3] = 6 * x[0]

    A[4 * x.size - 5][4 * x.size - 6] = 2
    A[4 * x.size - 5][4 * x.size - 5] = 6 * x[g]

    #print(A)
    f = np.zeros(4 * (g))
    for i in range(g):
        j = i + 1
        f[4 * i + 0] = y[i]
        f[4 * i + 1] = y[j]
        f[4 * i + 2] = 0
        f[4 * i + 3] = 0

    print(f)

    c = np.linalg.solve(A,f)
    spline = []
    for j in range(g):
        spline.append(np.poly1d((c[4*j+3], c[4*j+2], c[4*j+1], c[4*j])))
    #print(c)


    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    g = x.size-1
    A = np.zeros((4 * g, 4 * g), dtype = np.float64)
    for i in range(g):

        j = i + 1
        A[4 * i][4 * i + 0] = 1
        A[4 * i][4 * i + 1] = x[i]
        A[4 * i][4 * i + 2] = x[i]**2
        A[4 * i][4 * i + 3] = x[i]**3

        A[4 * i + 1][4 * i + 0] = 1
        A[4 * i + 1][4 * i + 1] = x[j]
        A[4 * i + 1][4 * i + 2] = x[j] ** 2
        A[4 * i + 1][4 * i + 3] = x[j] ** 3
####################
        if i < (g - 1):
            A[4 * i + 2][4 * i] = 0
            A[4 * i + 2][4 * i + 1] = 1
            A[4 * i + 2][4 * i + 2] = 2 * x[j]
            A[4 * i + 2][4 * i + 3] = 3 * x[j]**2

            A[4 * i + 2][4 * i + 4] = 0
            A[4 * i + 2][4 * i + 5] = -1
            A[4 * i + 2][4 * i + 6] = -2 * x[j]
            A[4 * i + 2][4 * i + 7] = -3 * x[j] ** 2
####################
            A[4 * i + 3][4 * i] = 0
            A[4 * i + 3][4 * i + 1] = 0
            A[4 * i + 3][4 * i + 2] = 2
            A[4 * i + 3][4 * i + 3] = 6 * x[j]

            A[4 * i + 3][4 * i + 4] = 0
            A[4 * i + 3][4 * i + 5] = 0
            A[4 * i + 3][4 * i + 6] = -2
            A[4 * i + 3][4 * i + 7] = -6 * x[j]

    A[4 * x.size - 6][0] = 0
    A[4 * x.size - 6][1] = 1
    A[4 * x.size - 6][2] = 2 * x[0]
    A[4 * x.size - 6][3] = 3 * x[0]**2

    A[4 * x.size - 6][4 * x.size - 8] = 0
    A[4 * x.size - 6][4 * x.size - 7] = -1
    A[4 * x.size - 6][4 * x.size - 6] = -2 * x[g]
    A[4 * x.size - 6][4 * x.size - 5] = -3 * x[g]**2

#################
    A[4 * x.size - 5][0] = 0
    A[4 * x.size - 5][1] = 0
    A[4 * x.size - 5][2] = 2
    A[4 * x.size - 5][3] = 6 * x[0]

    A[4 * x.size - 5][4 * x.size - 8] = 0
    A[4 * x.size - 5][4 * x.size - 7] = 0
    A[4 * x.size - 5][4 * x.size - 6] = -2
    A[4 * x.size - 5][4 * x.size - 5] = -6 * x[g]

#################

    f = np.zeros(4 * (g))
    for i in range(g):
        j = i + 1
        f[4 * i + 0] = y[i]
        f[4 * i + 1] = y[j]
        f[4 * i + 2] = 0
        f[4 * i + 3] = 0

    c = np.linalg.solve(A,f)
    spline = []
    for j in range(g):
        spline.append(np.poly1d((c[4*j+3], c[4*j+2], c[4*j+1], c[4*j])))

    return spline


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
