from collections import Counter

from nltk import word_tokenize
from nltk.lm.preprocessing import padded_everygrams

from utils.dataset import Dataset

LANGUAGE = 'german'

if __name__ == '__main__':
    data = Dataset(LANGUAGE)
    results = []
    sentences = set()
    for line in data.trainset:
        s = line['sentence']
        if s in sentences:
            continue
        sentences.add(s)
        for w in word_tokenize(s):
            result = padded_everygrams(2, w.lower())
            if result:
                results.extend(result)

    with open('datasets/{}/freq-ngrams.txt'.format(LANGUAGE), 'w') as out_file:
        out_file.write('ngram\tfreq\n')
        for k, v in Counter(results).most_common():
            if v == 1:
                break
            out_file.write('{}\t{}\n'.format(''.join(k), v))
