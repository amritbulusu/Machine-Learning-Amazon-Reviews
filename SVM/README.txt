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
SVM.py - contains the python program to create bag of words, train an SVM and predict it using test data. Prints the accuracy and confusion matrix.

			
OptdigitsExecution:
Install weka using the .exe file in the folder.
HowToExecuteWeka - Instructions to execute using Weka
optdigits_train.csv.ARFF - Converted to ARFF, the provided Optdigits training data. (First row contains indices)
optdigits_test.csv.ARFF - Converted to ARFF, the provided Optdigits test data. (First row contains indices)
nominaloptdigits_test.csv.ARFF - Converted class values to nominal.
nominaloptdigits_train.csv.ARFF - Converted class values to nominal.
weka-3-8-0jre-x64 - Weka installer





