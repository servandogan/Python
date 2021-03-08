import numpy as np
#import lib
import matplotlib as mpl
from matplotlib import image




def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    
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
    
    D = np.zeros((len(images), images[0].shape[1] * images[0].shape[0]))

    for i in range(len(images)):
        """flattenimage1 = images[i].flatten()
        D[i] = flattenimage1.flatten()"""
        D[i] = images[i].reshape(1, images[0].shape[1] * images[0].shape[0])

    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
 
    #mean_data = np.zeros((D.shape[1]))
    mean_data = np.average(D, axis=0)
#np.average(matrix, axis=0)
    mean_data1 = D - mean_data



    U, sig, V_t = np.linalg.svd(mean_data1, full_matrices=False)

    return V_t, sig, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:

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
  

    coefficients = np.zeros((len(images), pcs.shape[0]))

    for i in range(len(images)):
        normalizedimg = images[i].flatten() - mean_data
        coefficients[i] = np.dot(pcs, normalizedimg)

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):


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
