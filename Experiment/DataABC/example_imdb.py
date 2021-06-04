import tensorflow_datasets as tfds
from .abc_data import DataABC

BUFFER_SIZE = 10000
BATCH_SIZE = 64


class IMDB(DataABC):
    def __init__(self):
        dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                                as_supervised=True)
        train_examples, test_examples = dataset['train'], dataset['test']

        self.encoder = info.features['text'].encoder
        print('Vocabulary size: {}'.format(self.encoder.vocab_size))

        self.train_dataset = (train_examples
                                .shuffle(BUFFER_SIZE)
                                .padded_batch(BATCH_SIZE))

        self.test_dataset = (test_examples
                                .padded_batch(BATCH_SIZE))

    def _check_tokenize(self):
        sample_string = 'Hello TensorFlow.'
        encoded_string = self.encoder.encode(sample_string)
        print('Encoded string is {}'.format(encoded_string))

        original_string = self.encoder.decode(encoded_string)
        print('The original string: "{}"'.format(original_string))

        assert original_string == sample_string

        for index in encoded_string:
            print('{} ----> {}'.format(index, self.encoder.decode([index])))


    def get_train(self):  #=> dict or stream
        return {
            'train' : self.train_dataset,
            'test' : self.test_dataset
        }

    def get_eval(self):  #=> dict or stream
        return self.test_dataset


if __name__ == "__main__":
    imdb = IMDB()
    imdb._check_tokenize()

