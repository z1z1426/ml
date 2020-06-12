import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer


def demo():
    X = [[20, 3],
         [23, 7],
         [31, 10],
         [42, 13],
         [50, 7],
         [60, 5]]
    y = [0, 1, 1, 1, 0, 0]
    lr = linear_model.LogisticRegression()
    lr.fit(X, y)
    testX = [[28, 8]]
    label = lr.predict(testX)
    print(label)
    prob = lr.predict_proba(testX)
    print(prob)


def demo1():
    df = pd.read_csv('SMSSpamCollection.txt', delimiter='\t', header=None)
    y, X_train = df[0], df[1]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train)
    lr = linear_model.LogisticRegression()
    lr.fit(X, y)
    testX = vectorizer.transform(['URGENT! Your mobile No.1234 was awarded a Prize',
                                  "Hey honey, what's up"])
    predictions = lr.predict(testX)
    print(predictions)


if __name__ == '__main__':
    demo1()
    print(1)