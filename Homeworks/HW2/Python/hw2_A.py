from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn import metrics
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

if __name__ == '__main__':
    trainData = datasets.load_files('./Problem1/train')
    testData = datasets.load_files('./Problem1/test')

    vectorizer = CountVectorizer(stop_words=load_stop_words('./Problem1/stopwords.txt'), analyzer='word', token_pattern=u'(?u)\\b\\w+\\b')
    #vectorizer = CountVectorizer(analyzer='word', token_pattern=u'(?u)\\b\\w+\\b')

    vectorizer.fit(trainData.data)

    trainVector = vectorizer.transform(trainData.data)
    trainLabel = trainData.target

    testVector = vectorizer.transform(testData.data)
    testLabel = testData.target

    clf = naive_bayes.MultinomialNB()
    clf.fit(trainVector,trainLabel)

    coefrange = clf.feature_log_prob_ .shape[1]



    sortedcoefids_pos = sorted(range(coefrange), key=lambda i: clf.feature_log_prob_[0][i])[-5:]
    print('Top five most frequent words of neg (L=0) class:')
    for id in sortedcoefids_pos:
        print(vectorizer.get_feature_names()[id])

    sortedcoefids_neg = sorted(range(coefrange), key=lambda i: clf.feature_log_prob_[1][i])[-5:]
    print('Top five most frequent words of pos (L=0) class:')
    for id in sortedcoefids_neg:
        print(vectorizer.get_feature_names()[id])

    pred = clf.predict(testVector)
    print(metrics.confusion_matrix(testLabel,pred))