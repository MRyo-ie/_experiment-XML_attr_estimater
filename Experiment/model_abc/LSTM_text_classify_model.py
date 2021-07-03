
from .abc_model import ModelABC

import torch
import torch.nn as nn


class LSTM_TextClassifier_ptModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        # <pad>の単語IDが0なので、padding_idx=0としている
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # batch_first=Trueが大事！
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        #embeds.size() = (batch_size × len(sentence) × embedding_dim)
        _, lstm_out = self.lstm(embeds)
        # lstm_out[0].size() = (1 × batch_size × hidden_dim)
        tag_space = self.hidden2tag(lstm_out[0])
        # tag_space.size() = (1 × batch_size × tagset_size)

        # (batch_size × tagset_size)にするためにsqueeze()する
        tag_scores = self.softmax(tag_space.squeeze())
        # tag_scores.size() = (batch_size × tagset_size)

        return tag_scores  

    def load_weights(self, load_m_path='_logs/test/LSTM_classifier.pth',):
        if load_m_path is not None:
            param = torch.load(load_m_path)
            self.load_state_dict(param)
            print(f'[info] {load_m_path} loaded !')

    def save(self, save_f_path='_logs/test/LSTM_classifier.pth',):
        torch.save(self.state_dict(), save_f_path)




import tensorflow as tf

class LSTM_TextClassify_tfModel(ModelABC):
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

