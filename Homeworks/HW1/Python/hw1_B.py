from sklearn import linear_model
import numpy
from scipy.sparse import csr_matrix
import os


def loadData(path):
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(file,'rt') as review:



if __name__ == '__main__':
    loadData('./test/neg');
