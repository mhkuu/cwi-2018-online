from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def train(self, train_set):
        pass

    @abstractmethod
    def test(self, test_set):
        pass
