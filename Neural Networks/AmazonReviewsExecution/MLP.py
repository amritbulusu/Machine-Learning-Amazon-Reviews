from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
#import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def bag_of_words():
    train = pd.read_csv("train_dummy.csv", delimiter=',')
    num_reviews = train["review"].size

    #print "Cleaning and parsing the training set movie reviews...\n"
    clean_train_reviews = []
    for i in xrange(0, num_reviews):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    #print "Creating the bag of words..\n"

    #vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,  min_df=1, max_features=500)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    #print train_data_features
    train_data_features = train_data_features.toarray()
    scaler = StandardScaler()
    scaler.fit(train_data_features)
    train_data_features = scaler.transform(train_data_features)

    clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(500, ))

    clf = clf.fit(train_data_features, train["rating"])

    clean_test_reviews = []

    test = pd.read_csv("test_dummy.csv", delimiter=',')
    num_reviews1 = test["review"].size

    #print "Cleaning and parsing the test set movie reviews...\n"
    clean_train_reviews = []
    for i in xrange(0, num_reviews1):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    test_data_features = scaler.transform(test_data_features)
    #print len(test_data_features)

    #print "Predicting test labels ...\n"
    result = clf.predict(test_data_features)
    check = test['rating'].values
    #print train_data_features
    #print test_data_features
    #print "The accuracy score for decision tree is"
    print accuracy_score(check, result)
    print confusion_matrix(check, result)
    print classification_report(check, result)
    '''y, predicted = check, result
    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()'''

    #output = numpy.column_stack((train_data_features, train['rating']))
    #numpy.savez('output_test.csv', output, delimiter=',')

bag_of_words()


