from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib

# load data
train_url = "https://raw.githubusercontent.com/uvanlp/iml-2022/main/data/SST2/train.tsv"
train_df = pd.read_csv(train_url, sep='\t', header=0)
test_url = "https://raw.githubusercontent.com/uvanlp/iml-2022/main/data/SST2/test.tsv"
test_df = pd.read_csv(test_url, sep='\t', header=0)

x_train = train_df.sentence
y_train = train_df.label
x_test = test_df.sentence
y_test = test_df.label


# Before training the classifier, we convert the training and test data into TF-IDF representations.
vectorizer = TfidfVectorizer(lowercase=False)
# convert the training and test data into TF-IDF representations
train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)


# Build a RandomForestClassifier with 50 trees
rf = RandomForestClassifier(n_estimators = 50)
# fit the model with training data
rf.fit(train_vectors, y_train)
# get model predictions on the test set and report the prediction accuracy
pred = rf.predict(test_vectors)
print('Test accuracy:', accuracy_score(y_test, pred))


# pack the vectorizer and model
clf = make_pipeline(vectorizer, rf)
# save model to disk
joblib.dump(clf, 'random_forest_clf.pkl')
