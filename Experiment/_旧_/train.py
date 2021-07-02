
from ModelABC.LSTM_text_classify_model import LSTM_TextClassify_Model
from DataABC.example_imdb import IMDB
from abc_experiment import ExperimentABC



class Text_Classification_ExpTrain(ExperimentABC):
    """
    Train の抽象化
    """
    def exec(self):
        """
        モデルを学習（train）する。
        
        例）
            for m in models:
                self.model.fit(X, params['treat'], y)
        """
        ##<<--  train  -->>##
        for exp in self.experiments:
            exp_name = exp['exp_name']
            dataABC = exp['dataABC']
            model = exp['modelABC']

            print(f'[Info] Experiment： {exp_name} (Train)')
            model.fit(dataABC)





if __name__ == "__main__":
    imdb = IMDB()
    imdb._check_tokenize()

    exps = [
        {
            'exp_name' : 'LSTM <- imdb', 
            'dataABC' : imdb, 
            'modelABC' : LSTM_TextClassify_Model(imdb.encoder.vocab_size)
        }
    ]

    exp_train = Text_Classification_ExpTrain(exps)
    exp_train.exec()

