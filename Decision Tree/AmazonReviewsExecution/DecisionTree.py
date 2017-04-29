from sklearn.tree import DecisionTreeClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

def bag_of_words():
    train = pd.read_csv("train_dummy.csv", delimiter=',')
    num_reviews = train["review"].size

    print "Cleaning and parsing the training set movie reviews...\n"
    clean_train_reviews = []
    for i in xrange(0, num_reviews):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    print "Creating the bag of words..\n"

    #vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,  min_df=1)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    #print train_data_features
    train_data_features = train_data_features.toarray()

    clf = DecisionTreeClassifier(criterion="gini", splitter="best")

    clf = clf.fit(train_data_features, train["rating"])

    clean_test_reviews = []

    test = pd.read_csv("test_dummy.csv", delimiter=',')
    num_reviews1 = test["review"].size

    print "Cleaning and parsing the test set movie reviews...\n"
    clean_train_reviews = []
    for i in xrange(0, num_reviews1):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    #print len(test_data_features)
    tree.export_graphviz(clf, out_file='tree.dot')

    print "Predicting test labels ...\n"
    result = clf.predict(test_data_features)
    check = test['rating'].values
    #print train_data_features
    #print test_data_features
    print "The accuracy score is"
    print accuracy_score(check, result)

    #output = numpy.column_stack((train_data_features, train['rating']))
    #numpy.savez('output_test.csv', output, delimiter=',')

bag_of_words()



