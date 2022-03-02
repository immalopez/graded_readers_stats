##############################################################################
#                          LOGISTIC REGRESSION
##############################################################################
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def logit(X, y):
    solver = "saga"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y, test_size=0.2
    )
    print(np.unique(y).shape)

    return 0

    # data_train, data_test = train_test_split(data,
    #                                          test_size=0.2,
    #                                          random_state=42)
    # vectorizer = CountVectorizer()
    # vectorizer.fit(data_train[COL_RAW_TEXT])
    # x_train = vectorizer.transform(data_train[COL_RAW_TEXT])
    # x_test = vectorizer.transform(data_test[COL_RAW_TEXT])
    # y_train = data_train[COL_LEVEL]
    # y_test = data_test[COL_LEVEL]
    #
    # model = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_C)
    # model.fit(X_train, y_train)
    #
    # scores = cross_val_score(model, X_train, y_train, cv=CV_SPLITS)
    # print('=== BOW ===')
    # print('Cross-validation scores:', scores, sep='\n')
    # print('Cross-validation avg. score:', scores.mean(), sep='\n')
    # print('Model train score:', model.score(X_train, y_train), sep='\n')
    # print('Model test score:', model.score(X_test, y_test), sep='\n')
