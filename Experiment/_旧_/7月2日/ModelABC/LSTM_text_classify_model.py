
from .abc_model import ModelABC
import tensorflow as tf


class LSTM_TextClassify_Model(ModelABC):
    def __init__(self, vocab_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
    
    def fit(self, dataset):
        imdb = dataset.get_train()
        train_dataset = imdb['train']
        test_dataset = imdb['test']

        self.history = self.model.fit(train_dataset, epochs=10,
                        validation_data=test_dataset, 
                        validation_steps=30)

    def predict(self, data) -> int:
        """
        モデルで予測（predict）する。

        :return np.ndarray : model の予測結果(array)
        【実装例】
            score_arr = 2 * self.predict_proba(data)[:,1] - 1
            return np.array(score_arr)
        """
        raise NotImplementedError()

    def predict_orgf(self, data):
        raise NotImplementedError()
