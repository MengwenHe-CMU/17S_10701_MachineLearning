from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from matplotlib import pyplot
from matplotlib import gridspec
import numpy

class IonosphereData:
    g_features = list()
    b_features = list()
    def __init__(self):
        self.g_features=list()
        self.b_features=list()

def load_data(filename):
    with open(filename, 'rt') as stopWordsFile:
        data = IonosphereData()
        for line in stopWordsFile:
            feature = line.strip().split(',')
            if(feature[34]=='g'):
                featurevalues = list()
                for id in range(34):
                    featurevalues.append(float(feature[id]))
                data.g_features.append(featurevalues)
            else:
                featurevalues = list()
                for id in range(34):
                    featurevalues.append(float(feature[id]))
                data.b_features.append(featurevalues)
        return data

class IonosphereDataCV:
    train = IonosphereData()
    test = IonosphereData()
    def __init__(self):
        self.train.g_features=list()
        self.train.b_features=list()
        self.test.g_features=list()
        self.test.b_features=list()

def split_data(data, rate):
    datacv = IonosphereDataCV()

    gn = len(data.g_features)
    grand = numpy.random.rand(gn)
    gsortedids = sorted(range(gn), key=lambda i: grand[i])
    for id in gsortedids[:int(gn*rate)]:
        datacv.train.g_features.append(data.g_features[id])
    for id in gsortedids[-(gn-int(gn*rate)):]:
        datacv.test.g_features.append(data.g_features[id])

    bn = len(data.b_features)
    brand = numpy.random.rand(bn)
    bsortedids = sorted(range(bn), key=lambda i: brand[i])
    for id in bsortedids[:int(bn * rate)]:
        datacv.train.b_features.append(data.b_features[id])
    for id in bsortedids[-(bn - int(bn * rate)):]:
        datacv.test.b_features.append(data.b_features[id])
    return datacv


if __name__ == '__main__':
    data = load_data('./Problem2/ionosphere.txt')
    datacv = split_data(data,0.8)

    adaboost = list()
    adaboost.append(AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=1)))
    adaboost.append(AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)))
    adaboost.append(AdaBoostClassifier(n_estimators=100, base_estimator=BernoulliNB()))
    adaboost.append(AdaBoostClassifier(n_estimators=100, base_estimator=LogisticRegression()))
    adaboost.append(AdaBoostClassifier(n_estimators=100, base_estimator=svm.SVC(probability=True, kernel='linear')))
    adaboost.append(AdaBoostClassifier(n_estimators=100, base_estimator=svm.SVC(probability=True, kernel='rbf')))

    weakestimatornames = ["DecisionTreeClassifier (max_depth=4)", "DecisionStumpClassifier", "BernoulliNB", "LogisticRegression", "Linear SVM", "RBF SVM"]

    trainfeatures = datacv.train.g_features + datacv.train.b_features
    gtrainlen = len(datacv.train.g_features)
    btrainlen = len(datacv.train.b_features)
    trainlabel = numpy.ones(gtrainlen).tolist() + numpy.zeros(btrainlen).tolist()

    testfeatures = datacv.test.g_features + datacv.test.b_features
    gtestlen = len(datacv.test.g_features)
    btestlen = len(datacv.test.b_features)
    testlabel = numpy.ones(gtestlen).tolist() + numpy.zeros(btestlen).tolist()

    fig = pyplot.figure()
    ax = list()
    ax.append(fig.add_subplot(2, 3, 1))
    ax.append(fig.add_subplot(2, 3, 2))
    ax.append(fig.add_subplot(2, 3, 3))
    ax.append(fig.add_subplot(2, 3, 4))
    ax.append(fig.add_subplot(2, 3, 5))
    ax.append(fig.add_subplot(2, 3, 6))

    for id in range(6):
        print(id)

        adaboost[id].fit(trainfeatures, trainlabel)
        train_err = list()
        for i,y_pred in enumerate(adaboost[id].staged_predict(trainfeatures)):
            train_err.append(metrics.zero_one_loss(trainlabel,y_pred))
        print(1-min(train_err))
        test_err = list()
        for i, y_pred in enumerate(adaboost[id].staged_predict(testfeatures)):
            test_err.append(metrics.zero_one_loss(testlabel, y_pred))
        print(1-min(test_err))

        ax[id].set_title(weakestimatornames[id])
        ax[id].set_xlabel("n_estimators")
        ax[id].set_ylabel("error")
        ax[id].plot(numpy.arange(100) + 1, train_err,
                label='Train Error',
                color='blue')
        ax[id].plot(numpy.arange(100) + 1, test_err,
                label='Test Error',
                color='red')

        ax[id].legend()

    pyplot.show()