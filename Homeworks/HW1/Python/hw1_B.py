from time import time
import numpy
from sklearn import datasets
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.utils.extmath import density
import pickle



def load_stop_words(filename):
    with open(filename, 'rt') as stopWordsFile:
        stopWords = list()
        for line in stopWordsFile:
            word = line.strip()
            stopWords.append(word)
        return stopWords


if __name__ == '__main__':
    vectorizer = HashingVectorizer(stop_words=load_stop_words('./stopwords.txt'), non_negative=True, n_features=100000)

    trainData = datasets.load_files('./train')
    trainVector = vectorizer.fit_transform(trainData.data)
    trainLabel = trainData.target

    testData = datasets.load_files('./test')
    testVector = vectorizer.fit_transform(testData.data)
    testLabel = testData.target

    alphas = [1000.0,100.0,10.0,1.0,0.1,0.01]

    print('start ridge')
    for value in alphas:
        reg = linear_model.Ridge(alpha=value)

        print("start training")
        t0 = time()
        reg.fit(trainVector,trainLabel)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        pickle.dump(reg,open("ridge_"+str(value)+".sav","wb"))

        # reg = pickle.load(open("ridge_model.sav","rb"))

        pred = reg.predict(testVector)
        accuracy = ((pred - testLabel) ** 2).sum()
        print("accuracy:    %0.3f" % accuracy)

        r2score = reg.score(testVector, testLabel)
        print("R^2 score:   %0.3f" % r2score)

    print('start lasso')
    for value in alphas:
        reg = linear_model.Lasso(alpha=value)

        print("start training")
        t0 = time()
        reg.fit(trainVector,trainLabel)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        pickle.dump(reg,open("lasso_"+str(value)+".sav","wb"))

        # reg = pickle.load(open("ridge_model.sav","rb"))

        pred = reg.predict(testVector)
        accuracy = ((pred-testLabel)**2).sum()
        print("accuracy:    %0.3f" % accuracy)

        r2score = reg.score(testVector, testLabel)
        print("R^2 score:   %0.3f" % r2score)

