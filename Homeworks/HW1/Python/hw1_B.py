from time import time
import numpy
from sklearn import datasets
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.utils.extmath import density



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

    reg = linear_model.RidgeCV(1)
    t0 = time()
    reg.fit(trainVector,trainLabel)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = reg.predict(testVector)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score =metrics.accuracy_score(testLabel,pred)
    print("accuracy:   %0.3f" % score)

    print("dimensionality: %d" % reg.coef_.shape[1])
    print("density: %f" % density(reg.coef_))

    feature_names=vectorizer.get_feature_names()
    print("top 5 keywords per class:")
    for i, label in enumerate(trainData.target_names):
        top5 = numpy.argsort(reg.coef_[i])[-5:]
        print("%s: %s" % (label, " ".join(feature_names[top5])))
