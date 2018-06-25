import random
import numpy as np

from noesis.dataset.vocabulary import Vocabulary
from noesis.dataset import utils


class Dataset(object):
    """
    A class that encapsulates a dataset.

    Warning:
        Do not use this constructor directly, use one of the class methods to initialize.

    Note:
        Source or target sequences that are longer than the respective
        max length will be filtered.

    Args:
        max_len (int): maximum source sequence length
    """

    def __init__(self):
        # Declare vocabulary objects
        self.vocab = None
        self.data = None


    @classmethod
    def from_file(cls, path, vocab=None, max_vocab=50000):
        """
        Initialize a dataset from the file at given path. The file
        must contains a list of TAB-separated pairs of sequences.

        Note:
            Source or target sequences that are longer than the respective
            max length will be filtered.
            As specified by maximum vocabulary size, source and target
            vocabularies will be sorted in descending token frequency and cutoff.
            Tokens that are in the dataset but not retained in the vocabulary
            will be dropped in the sequences.

        Args:
            path (str): path to the dataset file
            vocab (Vocabulary): pre-populated Vocabulary object or a path of a file containing words for the source language, default `None`. If a pre-populated Vocabulary object, `src_max_vocab` wouldn't be used.
            max_vocab (int): maximum source vocabulary size
        """
        obj = cls()
        pairs = utils.prepare_data(path)
        return cls._encode(obj, pairs, vocab, max_vocab)

    def _encode(self, pairs, vocab=None, max_vocab=500000):
        """
        Encodes the source and target lists of sequences using source and target vocabularies.

        Note:
            Source or target sequences that are longer than the respective
            max length will be filtered.
            As specified by maximum vocabulary size, source and target
            vocabularies will be sorted in descending token frequency and cutoff.
            Tokens that are in the dataset but not retained in the vocabulary
            will be dropped in the sequences.

        Args:
            pairs (list): list of tuples (source sequences, target sequence)
            vocab (Vocabulary): pre-populated Vocabulary object or a path of a file containing words for the source language,
            default `None`. If a pre-populated Vocabulary object, `src_max_vocab` wouldn't be used.
            max_vocab (int): maximum source vocabulary size
        """
        # Read in vocabularies
        self.vocab = self._init_vocab(pairs, max_vocab, vocab)

        # Translate input sequences to token ids
        self.data = []
        for (context, candidates), target in pairs:
            c = self.vocab.indices_from_sequence(context)
            r = []
            for candidate in candidates:
                r.append(self.vocab.indices_from_sequence(candidate))
            self.data.append(((c, r), target))
        return self

    def _init_vocab(self, data, max_num_vocab, vocab):
        resp_vocab = Vocabulary(max_num_vocab)
        if vocab is None:
            for (context, candidates), target in data:
                resp_vocab.add_sequence(context)
                for candidate in candidates:
                    resp_vocab.add_sequence(candidate)
            resp_vocab.trim()
        elif isinstance(vocab, Vocabulary):
            resp_vocab = vocab
        elif isinstance(vocab, str):
            for tok in utils.read_vocabulary(vocab, max_num_vocab):
                resp_vocab.add_token(tok)
        else:
            raise AttributeError('{} is not a valid instance on a vocabulary. None, instance of Vocabulary class \
                                 and str are only supported formats for the vocabulary'.format(vocab))
        return resp_vocab

    def _pad(self, data):
        c = [pair[0][0] for pair in data]
        r = [pair[0][1] for pair in data]
        context = np.zeros([len(c), max([len(entry) for entry in c])], dtype=int)
        context.fill(self.vocab.PAD_token_id)
        context_lengths = np.zeros(len(c), dtype=int)

        for i, entry in enumerate(c):
            context[i, :len(entry)] = entry
            context_lengths[i] = len(entry)

        responses = np.zeros([len(r), max([len(entry) for entry in r]), max([len(cand) for entry in r for cand in entry])], dtype=int)
        responses.fill(self.vocab.PAD_token_id)
        responses_lengths = np.zeros([len(r), max([len(entry) for entry in r])], dtype=int)

        for i, entry in enumerate(r):
            for j, cand in enumerate(entry):
                responses[i, j, :len(cand)] = cand
                responses_lengths[i, j] = len(cand)

        return context, responses, context_lengths, responses_lengths

    def __len__(self):
        return len(self.data)

    def num_batches(self, batch_size):
        """
        Get the number of batches given batch size.

        Args:
            batch_size (int): number of examples in a batch

        Returns:
            (int) : number of batches
        """
        return len(range(0, len(self.data), batch_size))

    def make_batches(self, batch_size):
        """
        Create a generator that generates batches in batch_size over data.

        Args:
            batch_size (int): number of pairs in a mini-batch

        Yields:
            (list (str), list (str)): next pair of source and target variable in a batch
        """
        if len(self.data) < batch_size:
            raise OverflowError("batch size = {} cannot be larger than data size = {}".
                                format(batch_size, len(self.data)))
        for i in range(0, len(self.data), batch_size):
            cur_batch = self.data[i:i + batch_size]
            context, responses, context_lengths, responses_lengths = self._pad(cur_batch)
            target = np.asarray([pair[1] for pair in cur_batch])

            yield (context, responses, target, context_lengths, responses_lengths)

    def shuffle(self, seed=None):
        """
        Shuffle the data.

        Args:
            seed (int): provide a value for the random seed; default seed=None is truly random
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)