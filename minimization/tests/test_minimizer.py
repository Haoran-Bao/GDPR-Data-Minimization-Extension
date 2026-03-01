import numpy as np

from minimization import GeneralizeToRepresentative
from sklearn.tree import DecisionTreeClassifier


def test_minimizer_params():
    # Assume two features, age and height, and boolean label
    cells = [{"id": 1, "ranges": {"age": {"start": None, "end": 38}, "height": {"start": None, "end": 170}}, "label": 0,
              "representative": {"age": 26, "height": 149}},
             {"id": 2, "ranges": {"age": {"start": 39, "end": None}, "height": {"start": None, "end": 170}}, "label": 1,
              "representative": {"age": 58, "height": 163}},
             {"id": 3, "ranges": {"age": {"start": None, "end": 38}, "height": {"start": 171, "end": None}}, "label": 0,
              "representative": {"age": 31, "height": 184}},
             {"id": 4, "ranges": {"age": {"start": 39, "end": None}, "height": {"start": 171, "end": None}}, "label": 1,
              "representative": {"age": 45, "height": 176}}
             ]
    features = ['age', 'height']
    # numpy arrays do not support mixed types, unless they are structured arrays
    # so seems that categorical to one hot mapping should be done outside...
    X = np.array([[23, 165],
                     [45, 158],
                     [18, 190]])
    print(X.dtype)
    y = [1,1,0]
    base_est = DecisionTreeClassifier()
    base_est.fit(X, y)

    gen = GeneralizeToRepresentative(base_est, features=features, cells=cells)
    gen.fit()
    transformed = gen.transform(X)
    print(transformed)

def test_minimizer_fit():
    features = ['age', 'height']
    # numpy arrays do not support mixed types, unless they are structured arrays
    # so seems that categorical to one hot mapping should be done outside...
    X = np.array([[23, 165],
                  [45, 158],
                  [56, 123],
                  [67, 154],
                  [45, 149],
                  [42, 166],
                  [73, 172],
                  [94, 168],
                  [69, 175],
                  [24, 181],
                  [18, 190]])
    print(X)
    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    base_est = DecisionTreeClassifier()
    base_est.fit(X, y)
    predictions = base_est.predict(X)

    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    print(X)
    print(transformed)


