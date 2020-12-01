import numpy as np
#import lib
import matplotlib as mpl
from matplotlib import image




def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not nxn")

    if epsilon == -1.0:
        epsilon = np.finfo(float).eps
    z = True
    while z:
        x = np.random.randn(M.shape[0])
        for k in range(M.shape[0]):
            a = 0
            for i in range(M.shape[0]):
                a = a + x[i] * M[k][i]
                z = False
            if a == 0:
                z = True

    # Initialize residual list and residual of current eigenvector estimate
    residuals = []
    residual = 2.0 * epsilon
    anzahl = 0
    # Perform power iteration
    while residual > epsilon:
        Ax = np.dot(M, x)               #haben vektor x und rechnen es hoch zu x_n

        tmp = 0
        for i in range(x.shape[0]):
           tmp += x[i]**2
        tmp = np.sqrt(tmp)
        tmp2 = 0
        for i in range(x.shape[0]):
            tmp2 += Ax[i]**2
        tmp2 = np.sqrt(tmp2)

        residual = np.arccos(np.clip(np.dot(Ax, x)/(tmp * tmp2),-1.0,1.0,out = None))
        """residual = np.arccos(np.dot(Ax, x) / np.linalg.norm(Ax) * np.linalg.norm(x))"""

        Ax_norm = 0                     #da wir Ax normieren müssen
        for zeile in range(Ax.shape[0]): #länge von Ax[i]
            Ax_norm += (Ax[zeile])**2     #normierter vektor wird gebildet
        Ax_norm = np.sqrt(Ax_norm)      # von Ax wird die wurzel noch genommen
        x = Ax/Ax_norm

        residuals.append(residual)
        anzahl += 1
    print("Anzahl der durchläufe: %s" % anzahl)
    print("Residuum nach durchlauf: %s" % residual)
    print("Epsilon: %s" % epsilon)


    return x, residuals


####################################################################################################
# Exercise 2: Eigenfaces

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    #images = np.asarray(mpl.image.imread(images.append(lib.list_directory(path)), file_ending))
    bildernamen = lib.list_directory(path)
    bildernamen.sort()
    for i in range(len(bildernamen)):
        if file_ending in bildernamen[i]:
            bildarray = mpl.image.imread(path + '/' + bildernamen[i])
            images.append(np.asarray(bildarray).astype(float))

    dimension_y = images[0].shape[0]
    dimension_x = images[0].shape[1]

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    D = np.zeros((len(images), images[0].shape[1] * images[0].shape[0]))

    for i in range(len(images)):
        """flattenimage1 = images[i].flatten()
        D[i] = flattenimage1.flatten()"""
        D[i] = images[i].reshape(1, images[0].shape[1] * images[0].shape[0])

    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """

    #mean_data = np.zeros((D.shape[1]))
    mean_data = np.average(D, axis=0)
#np.average(matrix, axis=0)
    mean_data1 = D - mean_data



    U, sig, V_t = np.linalg.svd(mean_data1, full_matrices=False)

    return V_t, sig, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """
    tmp = 0
    for i  in range(singular_values.shape[0]):
        tmp += singular_values[i]

    tmp2 = 0
    for k in range(singular_values.shape[0]):
        tmp2 += singular_values[k]
        if tmp * threshold <= tmp2:
            k += 1
            break

    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input  from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    coefficients = np.zeros((len(images), pcs.shape[0]))

    for i in range(len(images)):
        normalizedimg = images[i].flatten() - mean_data
        coefficients[i] = np.dot(pcs, normalizedimg)

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    print(pcs)
    imgs_test = []
    bildernamen = lib.list_directory(path_test)
    bildernamen.sort()
    for i in range(len(bildernamen)):
        if ".png" in bildernamen[i]:
            bildarray = mpl.image.imread(path_test + '/' + bildernamen[i])
            imgs_test.append(np.asarray(bildarray).astype(float))

    coeffs_test = np.zeros(coeffs_train.shape)
    coeffs_test = project_faces(pcs, imgs_test, mean_data)

    scores = np.zeros((len(coeffs_train),len(coeffs_test)))

    for i in range(coeffs_train.shape[0]):
        for j in range(coeffs_train.shape[1]):
            x = np.linalg.norm(coeffs_train[i][j])
            y = np.linalg.norm(coeffs_test[i][j])
            scores = np.arccos(np.clip(np.dot(coeffs_test,coeffs_train.transpose())/x * y, -1.0,1.0,out = None))



    return scores, imgs_test, coeffs_test


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
