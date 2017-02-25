from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn import metrics
from matplotlib import pyplot
from matplotlib import gridspec
import numpy

class IonosphereData:
    g_features = list()
    b_features = list()

def load_data(filename):
    with open(filename, 'rt') as stopWordsFile:
        data = IonosphereData()
        for line in stopWordsFile:
            feature = line.strip().split(',')
            if(feature[34]=='g'):
                data.g_features.append(feature[:34]).astype(float)
            else:
                data.b_features.append(feature[:34]).astype(float)
        return data

class IonosphereDataCV:
    train = IonosphereData()
    test = IonosphereData()

def split_data(data, rate):
    datacv = IonosphereDataCV()

    gn = len(data.g_features)
    grand = numpy.random.rand(gn)
    gsortedids = sorted(range(gn), key=lambda i: grand[i])
    gtrainids = gsortedids[:int(gn*rate)]
    gtestids = gsortedids[-(gn-int(gn*rate)):]
    for id in gtrainids:
        datacv.train.g_features.append(data.g_features[id])
    for id in gtestids:
        datacv.test.g_features.append(data.g_features[id])

    bn = len(data.b_features)
    brand = numpy.random.rand(bn)
    bsortedids = sorted(range(bn), key=lambda i: brand[i])
    btrainids = bsortedids[:int(bn * rate)]
    btestids = bsortedids[-(bn - int(bn * rate)):]
    for id in btrainids:
        datacv.train.b_features.append(data.b_features[id])
    for id in btestids:
        datacv.test.b_features.append(data.b_features[id])
    return datacv


if __name__ == '__main__':
    data = load_data('./Problem2/ionosphere.txt')
    datacv = split_data(data,0.8)

    adaboost = AdaBoostClassifier(base_estimator=Perceptron, n_estimators=100)
    trainfeatures = datacv.train.g_features+datacv.train.b_features
    gtrainlen=len(datacv.train.g_features)
    btrainlen=len(datacv.train.b_features)
    trainlabel = numpy.ones(gtrainlen).tolist()+numpy.zeros(btrainlen).tolist()
    adaboost.fit(trainfeatures, trainlabel)

    testfeatures = datacv.test.g_features + datacv.test.b_features
    gtestlen=len(datacv.test.g_features)
    btestlen=len(datacv.test.b_features)
    testlabel = numpy.ones(gtestlen).tolist()+numpy.zeros(btestlen).tolist()
    ada_discrete_err = list()
    for i, y_pred in enumerate(adaboost.staged_predict(testfeatures)):
        ada_discrete_err.append(metrics.zero_one_loss(testlabel,y_pred))

    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    ax.plot(numpy.arange(100) + 1, ada_discrete_err,
            label='Discrete AdaBoost Test Error',
            color='red')

    pyplot.show()