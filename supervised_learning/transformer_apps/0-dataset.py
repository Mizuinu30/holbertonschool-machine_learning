import tensorflow_datasets as tfds
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

class Dataset:
    def __init__(self):
        """
        Class constructor

        Sets the public instance attributes:
            data_train:
                contains the ted_hrlr_translate/pt_to_en
                    tf.data.Dataset train split, loaded as_supervised
            data_valid:
                contains the ted_hrlr_translate/pt_to_en
                    tf.data.Dataset validate split, loaded as_supervised
            tokenizer_pt:
                the Portuguese tokenizer created from the training set
            tokenizer_en:
                the English tokenizer created from the training set
        """
        self.data_train = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="train",
                                    as_supervised=True)
        self.data_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="validation",
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset

        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple (pt, en)
                pt is the tf.Tensor containing the Portuguese sentence
                en is the tf.Tensor containing the corresponding English sentence

        Returns:
            tokenizer_pt: the Portuguese tokenizer
            tokenizer_en: the English tokenizer
        """
        # Initialize the tokenizers
        tokenizer_pt = BertWordPieceTokenizer("neuralmind/bert-base-portuguese-cased-vocab.txt", lowercase=False)
        tokenizer_en = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

        # Train the tokenizers with a maximum vocabulary size of 2**13
        pt_sentences = [pt.numpy().decode('utf-8') for pt, en in data]
        en_sentences = [en.numpy().decode('utf-8') for pt, en in data]

        tokenizer_pt.train_from_iterator(pt_sentences, vocab_size=2**13)
        tokenizer_en.train_from_iterator(en_sentences, vocab_size=2**13)

        return tokenizer_pt, tokenizer_en
