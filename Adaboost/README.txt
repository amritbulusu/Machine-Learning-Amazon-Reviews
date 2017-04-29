AmazonReviewsExection:
Install Scikit and from NLTK Corpus, download Stopwords
From command line, run these commands.
python -m pip install scikit-learn
python -m nltk.downloader stopwords
Then run the file using
python MLP.py

or 

Run the script file - ml.sh

train_dummy - Dummy dataset created from the main amazon_baby_train by considering the first 8000 reviews.
test_dummy - Dummy dataset created from the main amazon_baby_test by considering the first 2000 reviews.
KaggleWord2VecUtility.py - From a given review, it converts the review to a string, removes the non-letters, coverts to 		lower-case, removes the stopwords and returns the remaining words.
DecisionTree.py - contains the python program to create bag of words, train a multilayer perceptron model and predict it using test data. Prints the accuracy, confusion matrix and precision,recall and F1 metrics.







