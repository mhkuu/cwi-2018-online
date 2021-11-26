from collections import Counter

from nltk import word_tokenize
from nltk.lm.preprocessing import padded_everygrams

from utils.dataset import Dataset

LANGUAGES = ['english', 'spanish', 'german', 'french']

if __name__ == '__main__':
    for language in LANGUAGES:
        data = Dataset(language)
        results = []
        for dataset in [data.devset, data.trainset, data.testset]:
            sentences = set()
            for line in dataset:
                s = line['sentence']
                if s in sentences:
                    continue
                sentences.add(s)
                for w in word_tokenize(s):
                    result = padded_everygrams(3, w.lower())
                    if result:
                        results.extend(result)

        with open('datasets/{}/freq-ngrams.txt'.format(language), 'w') as out_file:
            out_file.write('ngram\tfreq\n')
            for k, v in Counter(results).most_common():
                if v == 1:
                    break
                out_file.write('{}\t{}\n'.format(''.join(k), v))
