
from .abc_model import ModelABC
import tensorflow as tf


class BERT_TextClassify_Model(ModelABC):
    
    def __init__(self, vocab_size):
        self.classifier_model = self.build_classifier_model()
        self._check_model(self.classifier_model)
    
        self.epochs = 5
        self.steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        self.num_train_steps = steps_per_epoch * epochs
        self.num_warmup_steps = int(0.1*num_train_steps)

        self.init_lr = 3e-5
        self.history = None

        from official.nlp import optimization  # to create AdamW optmizer
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()

        self.classifier_model.compile(optimizer=optimizer,
                                        loss=loss,
                                        metrics=metrics)

    def _check_model(self, classifier_model):
        text_test = ['this is such an amazing movie!']

        bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
        text_preprocessed = bert_preprocess_model(text_test)
        print(f'Keys       : {list(text_preprocessed.keys())}')
        print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
        print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
        print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
        print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

        bert_raw_result = classifier_model(tf.constant(text_test))
        print(tf.sigmoid(bert_raw_result))

        tf.keras.utils.plot_model(classifier_model)

    def build_classifier_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)


    def fit(self, dataset):
        ds_train = dataset.get_train()
        print(f'Training model with {tfhub_handle_encoder}')
        self.history = self.classifier_model.fit(x=ds_train['train'],
                                                    validation_data=ds_train['valid'],
                                                    epochs=self.epochs)

    def plot_history(self):
        import matplotlib.pyplot as plt
        history_dict = self.history.history
        print(history_dict.keys())

        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

    def predict(self, data) -> float:
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
