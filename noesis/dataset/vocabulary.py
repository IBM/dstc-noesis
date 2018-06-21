class Vocabulary(object):
    """
    A Vocabulary stores a set of words belonging to a particular language. Words in the source vocabulary are mapped
    to unique integer IDs during encoding. Words in the target vocabulary are recovered from the model's output
    during decoding.
    In addition to the words in the actual language, a Vocabulary includes three reserved tokens (and IDs) for the
    start-of-sentence and end-of-sentence markers, and for a special 'mask' marker used to handle
    rare/unknown words.
    The Vocabulary is sorted in descending order based on frequency. If the number of words seen is greater than
    the maximum size of the Vocabulary, the remaining least-frequent words are ignored.

    Args:
        size (int): maximum number of words allowed in this vocabulary
    """
    def __init__(self, size):
        self.UNK_token_name = "__unk__"
        self.PAD_token_name = "__pad__"
        self.UNK_token_id = 0
        self.PAD_token_id = 1

        self._reserved = set([self.UNK_token_name, self.PAD_token_name])
        self._reserved_token_id = [(self.UNK_token_name, self.UNK_token_id),
                                   (self.PAD_token_name, self.PAD_token_id)]

        self.stoi = dict([(tok, idx) for tok, idx in self._reserved_token_id])
        self.itos = dict([(idx, tok) for tok, idx in self._reserved_token_id])

        self.stoc = {}

        self._num_tokens = 0
        self._num_reserved = 2

        self.sorted = False
        self.size = size

    def trim(self):
        """
        Sorts the vocabulary in descending order based on frequency
        """
        sorted_vocab_count = sorted(self.stoc.items(), key=lambda x: x[1], reverse=True)[:self.size]
        self.stoi = dict([(w, self._num_reserved + idx) for idx, (w, _) in enumerate(sorted_vocab_count)])
        self.itos = dict([(idx, w) for w, idx in self.stoi.items()])
        for tok, idx in self._reserved_token_id:
            self.stoi[tok] = idx
            self.itos[idx] = tok
        if self._num_tokens > self.size:
            self._num_tokens = self.size
        self.sorted = True

    def check_sorted(self):
        """
        Sorts the vocabulary (if it is not already sorted).
        """
        if not self.sorted:
            self.trim()

    def get_index(self, token):
        """
        Returns:
           ID (int): ID of the given token.
        """
        self.check_sorted()
        return self.stoi[token]

    def get_token(self, index):
        """
        Returns:
            token (str): token with ID equal to the given index.
        """
        self.check_sorted()
        return self.itos[index]

    def get_vocab_size(self):
        """
        Returns:
            size (int): maximum number of words in the vocabulary.
        """
        self.check_sorted()
        return self._num_tokens + self._num_reserved

    def add_token(self, token):
        """
        Adds an occurrence of a token to the vocabulary, incrementing its observed frequency if the word already exists.

        Args:
             token (int): word to add
        """
        if token in self._reserved:
            return
        if token not in self.stoc:
            self.stoc[token] = 1
            self._num_tokens += 1
        else:
            self.stoc[token] += 1
        self.sorted = False

    def add_sequence(self, sequence):
        """
        Adds a sequence of words to the vocabulary.

        Args:
             sequence(list(str)): list of words, e.g. representing a sentence.
        """
        for tok in sequence:
            self.add_token(tok)

    def indices_from_sequence(self, sequence):
        """
        Maps a list of words to their token IDs, or else the 'mask' token if the word is rare/unknown.

        Args:
            sequence (list(str)): list of words to map

        Returns:
            list(int): list of mapped IDs
        """
        self.check_sorted()
        return [self.stoi[tok]
                if tok in self.stoi
                else self.UNK_token_id
                for tok in sequence]

    def sequence_from_indices(self, indices):
        """
        Recover a sentence from a list of token IDs.

        Args:
            indices (list(int)): list of token IDs.

        Returns:
            list(str): recovered sentence, represented as a list of words
        """
        seq = [self.itos[idx] for idx in indices]
        return seq

    def save(self, file_name):
        """
        Writes this Vocabulary to disk in a pickle file.

        Args:
             file_name (str): path to the target pickle file
        """
        self.check_sorted()
        vocab_size = self.get_vocab_size()
        with open(file_name, "w") as f:
            for index, token in self.itos.items():
                if token in self._reserved:
                    continue
                if index < vocab_size -1:
                    f.write(token + "\n")
                else:
                    f.write(token)

    @classmethod
    def load(cls, file_name):
        """
        Loads a Vocabulary from a pickle file on disk.

        Args:
            file_name (str): path to the pickle file

        Returns:
            Vocabulary: loaded Vocabulary
        """
        with open(file_name, "r") as f:
            tokens = f.readlines()
        vocab = Vocabulary(len(tokens))
        for token in tokens:
            vocab.add_token(token.strip())
        return vocab

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        self.check_sorted()
        other.check_sorted()

        if self.stoc == other.stoc and self.stoi == other.stoi \
           and self.itos == other.itos:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.stoi)