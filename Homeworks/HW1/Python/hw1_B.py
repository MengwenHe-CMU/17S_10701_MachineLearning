from time import time
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from matplotlib import pyplot
from matplotlib import gridspec
import numpy

def load_stop_words(filename):
    with open(filename, 'rt') as stopWordsFile:
        stopWords = list()
        for line in stopWordsFile:
            word = line.strip()
            stopWords.append(word)
        return stopWords

def regression_train(reg, trainVector, trainLabel):
        print("start training")
        t0 = time()
        reg.fit(trainVector,trainLabel)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        return reg

def regression_test(reg,testVector,testLabel):
    pred = reg.predict(testVector)
    accuracy = 1 - ((pred - testLabel) ** 2).sum() / len(pred)
    print("accuracy:    %0.3f" % accuracy)
    return accuracy


if __name__ == '__main__':
    trainData = datasets.load_files('./train')
    testData = datasets.load_files('./test')

    vectorizer = CountVectorizer(stop_words=load_stop_words('./stopwords.txt'))
    vectorizer.fit(trainData.data+testData.data)

    trainVector = vectorizer.transform(trainData.data)
    trainLabel = trainData.target

    testVector = vectorizer.transform(testData.data)
    testLabel = testData.target

    C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    logC = numpy.log10(C)

    l2logisticaccuracytrain = list()
    l2logisticaccuracytest = list()
    l2logisticweightnorm = list()
    l2logisticsparsity = list()

    l1logisticaccuracytrain = list()
    l1logisticaccuracytest = list()
    l1logisticweightnorm = list()
    l1logisticsparsity =list()

    for value in C:
        l2logistic = regression_train(linear_model.LogisticRegression(C=value, penalty='l2'),trainVector,trainLabel)
        l2logisticaccuracytrain.append(regression_test(l2logistic,trainVector,trainLabel))
        l2logisticaccuracytest.append(regression_test(l2logistic,testVector,testLabel))
        l2logisticweightnorm.append(numpy.linalg.norm(l2logistic.coef_, ord=2))
        l2zeronum=l2logistic.coef_.shape[1]-numpy.count_nonzero(l2logistic.coef_)
        l2logisticsparsity.append(l2zeronum/l2logistic.coef_.shape[1])

        l1logistic = regression_train(linear_model.LogisticRegression(C=value, penalty='l1'),trainVector,trainLabel)
        l1logisticaccuracytrain.append(regression_test(l1logistic,trainVector,trainLabel))
        l1logisticaccuracytest.append(regression_test(l1logistic,testVector,testLabel))
        l1logisticweightnorm.append(numpy.linalg.norm(l1logistic.coef_,ord=1))
        l1zeronum = l1logistic.coef_.shape[1] - numpy.count_nonzero(l1logistic.coef_)
        l1logisticsparsity.append(l1zeronum / l1logistic.coef_.shape[1])

    fig1 = pyplot.figure(figsize=(10, 5))
    grid = gridspec.GridSpec(1, 2)

    l2logisticfig = fig1.add_subplot(grid[0, 0])
    l2logisticfig.plot(logC,l2logisticaccuracytest,'-o',color='red', label="Test")
    l2logisticfig.plot(logC,l2logisticaccuracytrain,'-o',color='blue', label="Train")
    l2logisticfig.set_xlabel("Log C")
    l2logisticfig.set_ylabel("Accuracy")
    l2logisticfig.legend()
    l2logisticfig.set_title("l2logistic Regression Accuracy")

    l1logisticfig = fig1.add_subplot(grid[0, 1])
    l1logisticfig.plot(logC, l1logisticaccuracytest, '-o', color='red', label="Test")
    l1logisticfig.plot(logC, l1logisticaccuracytrain, '-o', color='blue', label="Train")
    l1logisticfig.set_xlabel("Log C")
    l1logisticfig.set_ylabel("Accuracy")
    l2logisticfig.legend()
    l1logisticfig.set_title("l1logistic Regression Accuracy")

    fig1.savefig("result1.png", bbox_inches='tight')
    fig1.show()

    fig2 = pyplot.figure(figsize=(10, 5))
    grid = gridspec.GridSpec(1, 2)

    l2logisticnormfig = fig2.add_subplot(grid[0,0])
    l2logisticnormfig.plot(logC,l2logisticweightnorm, '-o', color='blue', label="L2")
    l2logisticnormfig.set_xlabel("Log C")
    l2logisticnormfig.set_ylabel("Weight L2 Norm")
    l2logisticnormfig.legend()
    l2logisticnormfig.set_title("l2logistic Regression Weight Norm")

    l1logisticnormfig = fig2.add_subplot(grid[0, 1])
    l1logisticnormfig.plot(logC, l1logisticweightnorm, '-o', color='red', label="L1")
    l1logisticnormfig.set_xlabel("Log C")
    l1logisticnormfig.set_ylabel("Weight L1 Norm")
    l1logisticnormfig.legend()
    l1logisticnormfig.set_title("l1logistic Regression Weight Norm")

    fig2.savefig("result2.png", bbox_inches='tight')
    fig2.show()

    fig3 = pyplot.figure(figsize=(10, 5))
    grid = gridspec.GridSpec(1,1)

    sparsityfig = fig3.add_subplot(grid[0, 0])
    sparsityfig.plot(logC, l2logisticsparsity, '-o', color='blue', label="L2")
    sparsityfig.plot(logC, l1logisticsparsity, '-o', color='red', label="L1")
    sparsityfig.set_xlabel("Log C")
    sparsityfig.set_ylabel("Sparsity")
    sparsityfig.legend()

    fig3.savefig("result3.png", bbox_inches='tight')
    fig3.show()

    l2logistic = regression_train(linear_model.LogisticRegression(C=0.01, penalty='l2'), trainVector, trainLabel)
    l2logisticrange = l2logistic.coef_.shape[1]
    sortedl2logisticcoefids = sorted(range(l2logisticrange), key=lambda i: l2logistic.coef_[0][i])

    print("Largest weight indexed words:")
    l2logisticids = sortedl2logisticcoefids[-5:]
    for id in l2logisticids:
        print(vectorizer.get_feature_names()[id])

    print("Least weight indexed words:")
    l2logisticids = sortedl2logisticcoefids[:5]
    for id in l2logisticids:
        print(vectorizer.get_feature_names()[id])

    predprob = l2logistic.predict_proba(testVector)
    negpredprobids = sorted(range(predprob.shape[0]), key=lambda i: predprob[i][0])
    pospredprobids = sorted(range(predprob.shape[0]), key=lambda i: predprob[i][1])

    print("most neg: %d" % negpredprobids[0])
    print(testData.filenames[negpredprobids[0]])
    print(testData.data[negpredprobids[0]])
    print("most pos: %d" % pospredprobids[0])
    print(testData.filenames[pospredprobids[0]])
    print(testData.data[pospredprobids[0]])
