from sklearn.dummy import DummyClassifier

from .abstract import AbstractModel


class Dummy(AbstractModel):

    def __init__(self):
        self.model = DummyClassifier(strategy="stratified")

    def train(self, train_set):
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

        return self.model.predict(X)
