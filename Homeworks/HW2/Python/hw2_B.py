from sklearn. import emsem
from sklearn import metrics

class Ionosphere(object):
    features = list()
    labels = list()

def load_data(filename):
    with open(filename, 'rt') as stopWordsFile:
        data = Ionosphere()
        for line in stopWordsFile:
            feature = line.strip().split(',')
            data.features.append(feature[:34])
            data.labels.append(feature[-1:])
        return data

if __name__ == '__main__':
    data = load_data('./Problem2/ionosphere.txt')

    bdt = Ada