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
    accuracy = ((pred - testLabel) ** 2).sum() / len(pred)
    print("accuracy:    %0.3f" % accuracy)
    r2score = reg.score(testVector, testLabel)
    print("R^2 score:   %0.3f" % r2score)
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

    alphas = [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01]
    logc = numpy.log10(numpy.ones_like(alphas)/alphas)

    ridgeaccuracy = list()
    ridgeweightnorm = list()
    ridgesparsity = list()
    lassoaccuracy = list()
    lassoweightnorm = list()
    lassosparsity =list()

    for value in alphas:
        ridge = regression_train(linear_model.Ridge(alpha = value),trainVector,trainLabel)
        ridgeaccuracy.append(regression_test(ridge,testVector,testLabel))
        ridgeweightnorm.append(numpy.linalg.norm(ridge.coef_, ord=2))
        nonzeronum=numpy.count_nonzero(ridge.coef_)
        coefnum=len(ridge.coef_)
        ridgesparsity.append((coefnum-nonzeronum)/coefnum)

        lasso = regression_train(linear_model.Lasso(alpha = value),trainVector,trainLabel)
        lassoaccuracy.append(regression_test(lasso,testVector,testLabel))
        lassoweightnorm.append(numpy.linalg.norm(lasso.coef_,ord=1))
        nonzeronum = numpy.count_nonzero(lasso.coef_)
        coefnum = len(lasso.coef_)
        lassosparsity.append((coefnum-nonzeronum)/coefnum)

    figure_width = 20
    figure_height = 20
    fig1 = pyplot.figure(figsize=(figure_width, figure_height))
    grid = gridspec.GridSpec(2, 2)

    ridgefig = fig1.add_subplot(grid[0, 0])
    ridgefig.scatter(logc,ridgeaccuracy,color='blue')
    ridgefig.set_xlabel("Log C")
    ridgefig.set_ylabel("Accuracy")
    ridgefig.set_title("Ridge Regression Accuracy")

    lassofig = fig1.add_subplot(grid[0, 1])
    lassofig.scatter(logc, lassoaccuracy, color='red')
    lassofig.set_xlabel("Log C")
    lassofig.set_ylabel("Accuracy")
    lassofig.set_title("Lasso Regression Accuracy")

    ridgenormfig = fig1.add_subplot(grid[1,0])
    ridgenormfig.scatter(logc,ridgeweightnorm, color='blue')
    ridgenormfig.set_xlabel("Log C")
    ridgenormfig.set_ylabel("Weight L2 Norm")
    ridgenormfig.set_title("Ridge Regression Weight Norm")

    lassonormfig = fig1.add_subplot(grid[1, 1])
    lassonormfig.scatter(logc, lassoweightnorm, color='red')
    lassonormfig.set_xlabel("Log C")
    lassonormfig.set_ylabel("Weight L1 Norm")
    lassonormfig.set_title("Lasso Regression Weight Norm")

    fig1.savefig("result1.png", bbox_inches='tight')
    fig1.show()

    fig2 = pyplot.figure(figsize=(figure_width, figure_height))
    grid = gridspec.GridSpec(1,1)

    sparsityfig = fig2.add_subplot(grid[0, 0])
    sparsityfig.plot(logc, ridgesparsity, '-', color='blue')
    sparsityfig.plot(logc, lassosparsity, '-', color='red')

    fig2.savefig("result2.png", bbox_inches='tight')
    fig2.show()

    ridge = regression_train(linear_model.Ridge(alpha=1000), trainVector, trainLabel)
    sortedridgecoefids = sorted(range(len(ridge.coef_)), key=lambda i: ridge.coef_[i])

    print("Largest weight indexed words:")
    ridgeids = sortedridgecoefids[-5:]
    for id in ridgeids:
        print(vectorizer.get_feature_names()[id])

    print("Least weight indexed words:")
    ridgeids = sortedridgecoefids[:5]
    for id in ridgeids:
        print(vectorizer.get_feature_names()[id])

    pred = ridge.predict(testVector)
    sortedpredids = sorted(range(len(pred)), key=lambda i: pred[i])
    ids = sortedpredids[-1:] + sortedpredids[:1]
    for id in ids:
        print(testData.data[id])