from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
#import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
#import matplotlib.pyplot as plt


def bag_of_words():
    train = pd.read_csv("train_dummy.csv", delimiter=',')
    num_reviews = train["review"].size

    #print "Cleaning and parsing the training set movie reviews...\n"
    clean_train_reviews = []
    for i in xrange(0, num_reviews):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    
    vectorizer = TfidfVectorizer(max_features=2500, min_df=4)
    train_data_features = vectorizer.fit_transform(clean_train_reviews).todense()

    print train_data_features.shape
    clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100),n_estimators=600, learning_rate=1.5)
    train_output = train['rating']
    clf = clf.fit(train_data_features, train_output)

    clean_test_reviews = []

    test = pd.read_csv("test_dummy.csv", delimiter=',')
    num_reviews1 = test["review"].size

    clean_train_reviews = []
    for i in xrange(0, num_reviews1):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    test_data_features = vectorizer.transform(clean_test_reviews).todense()

    result = clf.predict(test_data_features)
    check = test['rating']
    print check
    
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


