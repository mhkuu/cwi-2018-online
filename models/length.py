from .abstract import AbstractModel

LENGTH_CUTOFF = 8


class Length(AbstractModel):
    def __init__(self, length_cutoff):
        self.length_cutoff = length_cutoff

    def train(self, train_set):
        pass

    def test(self, test_set):
        result = []
        for sent in test_set:
            result.append(str(int(len(sent['target_word']) >= self.length_cutoff)))

        return result
