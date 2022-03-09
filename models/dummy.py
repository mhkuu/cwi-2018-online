from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError

from .abstract import AbstractModel


class Dummy(AbstractModel):

    def __init__(self):
        self.model = DummyClassifier(strategy="stratified")

    def train(self, train_set):
        if not train_set:
            return

        X = []
        y = []
        for sent in train_set:
            X.append(sent['gold_label'])
            y.append(sent['gold_label'])

        self.model.fit(X, y)

    def test(self, test_set):
        X = []
        for sent in test_set:
            X.append(sent['gold_label'])

        try:
            return self.model.predict(X)
        except NotFittedError:
            return False
