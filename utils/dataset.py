import csv


class Dataset:

    def __init__(self, language):
        self.language = language

        trainset_path = "datasets/{}/{}_Train.tsv".format(language, language.capitalize())
        devset_path = "datasets/{}/{}_Dev.tsv".format(language, language.capitalize())
        testset_path = "datasets/{}/{}_Test.tsv".format(language, language.capitalize())

        self.trainset = self.read_dataset(trainset_path)
        self.devset = self.read_dataset(devset_path)
        self.testset = self.read_dataset(testset_path)

    def read_dataset(self, file_path):
        try:
            with open(file_path) as file:
                fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                              'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
                reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

                # We're only interested in one-word forms -- remove the phrases
                dataset = [sent for sent in reader if len(sent['target_word'].split()) == 1]
        except OSError as exception:
            dataset = []

        return dataset


def load_data(language, verbose=False):
    data = Dataset(language)
    if verbose:
        print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))
    return data
