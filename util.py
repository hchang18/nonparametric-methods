# util.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def read_data(filename):
    file = open(filename)
    file_content = file.readlines()
    # clean up to make it into list
    file_content = [row.rstrip('\n').lstrip(' ').replace('  ', ' ').split(' ')
                    for row in file_content]
    # change into array (float)
    raw_data = np.array(file_content)
    data = raw_data[1:].astype(np.float)
    y = data[:, 0]
    x = data[:, 1]
    return data, x, y


def plot_2darray(X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot of X and Y')
    ax.scatter(X, Y)
    ax.grid(True)
    leg = mpatches.Patch(color=None, label='original data plots')
    ax.legend(handles=[leg])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()


