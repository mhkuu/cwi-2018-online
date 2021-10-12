def retrieve_ngram_freq(language, break_at=None):
    result = dict()
    with open('datasets/{}/freq-ngrams.txt'.format(language), 'r') as f:
        for n, line in enumerate(f):
            if n == 0:  # skip header
                continue
            if break_at and n == break_at:  # break at N ngrams
                break
            columns = line.strip('\n').split('\t')
            word = columns[0]
            freq = columns[1]
            result[word] = int(freq)

    return result
