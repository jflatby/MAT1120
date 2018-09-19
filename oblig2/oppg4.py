from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

data = np.array([
        [0.08, 4.05],
        [0.12, 4.15],
        [0.20, 3.85],
        [0.38, -0.22]
    ])

def least_squares_fit_1():
    #Design matrix
    X = np.insert(data, 0, 1, axis=1)[:, :-1]
    X = np.insert(X, 2, X[:, 1:].flatten()**2, axis=1)

    #y-values
    y = data[:, 1]

    X_T = np.transpose(X)

    left_side = np.dot(X_T, X)
    right_side = np.dot(X_T, y)

    #inverse left side and dot with right side

    B = np.dot(np.linalg.inv(left_side), right_side)

    x = np.linspace(0, 1, 1000)
    plt.plot(x, B[0] + B[1]*x + B[2]*x**2)
    plt.scatter(data[:, 0], data[:, 1])
    #plt.savefig("oppga.png")
    #plt.show()

    return B


def least_squares_fit_2():
    #Design matrix
    x_values = data[:, :-1]

    X = np.sin(2*np.pi*x_values)
    X = np.insert(X, 1, np.cos(2*np.pi*x_values).flatten(), axis=1)

    y_values = data[:, 1:]

    X_T = np.transpose(X)

    left_side = np.dot(X_T, X)
    right_side = np.dot(X_T, y_values)

    a_b = np.dot(np.linalg.inv(left_side), right_side)

    x = np.linspace(0, 1, 1000)
    plt.plot(x, a_b[0]*np.sin(2*np.pi*x) + a_b[1]*np.cos(2*np.pi*x))
    plt.scatter(data[:, 0], data[:, 1])
    #plt.savefig("oppgb.png")
    #plt.show()

    return a_b



def calculate_error(beta, a_b):
    #Beta-values from A:
    x_values = data[:, :-1]
    y_values = data[:, 1:]

    epsilon1 = y_values - (beta[0] + beta[1]*x_values + beta[2]*x_values**2)
    norm1 = np.linalg.norm(epsilon1)

    epsilon2 = y_values - (a_b[0]*np.sin(2*np.pi*x_values) + a_b[1]*np.cos(2*np.pi*x_values))
    norm2 = np.linalg.norm(epsilon2)

    print norm1, norm2


calculate_error(least_squares_fit_1(), least_squares_fit_2())