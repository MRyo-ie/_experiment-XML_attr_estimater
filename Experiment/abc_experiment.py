from abc import ABCMeta, abstractmethod
import os
import os.path as osp
import numpy as np
import pandas as pd

from ModelABC.abc_model import ModelABC
from DataABC.abc_data import DataPPP, DataABC



class ExperimentABC(metaclass=ABCMeta):
    """
    (Abstruct) Uplift Experiment Template.
    Uplift 検証実験 を効率的に扱えるようにするための Abstruct（抽象）クラス。

    【目的】
        ・ tarin, evaluate, test, ... の抽象化。
        ・ DeepLearning モデルとの統一。
        ・ SHAP（構造的には、ExperimentABC の拡張とみなせる）（引数：学習済みModel, Data）
        ・ 分散処理？（のための要素分析）
    
    【実験の構造】
        ExperimentABC[.train(), .evaluate()] 
            ← ( ModelABC[.fit(), .predict()],  DataABC[.get_train(), .get_eval()] )

    【Quick Start】
        ##<<--  Model:(Logistic, RF), DataABC:Basic  -->>##
        experiments = [
            {
                'exp_name' : 'logis_basic', 
                'dataABC' : DataABC_UpliftBasic(X_train, X_test, y_train, y_test, w_train, w_test), 
                'modelABC' : LogitReg_SingleModel()
            }, {
                'exp_name' : 'rf_basic', 
                'dataABC' : dataABC_base, 
                'modelABC' : RF_SingleModel()
            },
        ]
        ## Experiment Train
        exp_train_base = ExpTrain_UpliftBasic()
        exp_train_base.add_all(experiments)
        # または、
        # exp_train_base.add('logis', dataABC_base, model_logis)
        # exp_train_base.add('rf', dataABC_base, model_rf)
        exp_train_base.exec()

        ## Experiment Evalute
        exp_eval_base = ExpEvaluate_UpliftBasic(experiments)
        exp_eval_base.exec()
        result_s_abc = exp_eval_base.get_result_dict()
    """
    def __init__(self, experiments:list=None):
        """
        :param list experiments: [
                {
                    'exp_name' : exp_name,
                    'dataABC'  : data, 
                    'modelABC' : model,
                }, 
                {
                    ... 
                },
            ]
        """
        self.experiments = []
        if experiments is not None:
            self.add_all(experiments)

    def add(self, exp_name:str, data:DataABC, model:ModelABC):
        if not issubclass(type(model), ModelABC):
            raise Exception('[Error] ModelABC を継承していないモデルが入力されました。', type(model))
        if not issubclass(type(data), DataABC):
            raise Exception('[Error] DataABC を継承していないモデルが入力されました。:', type(data))

        self.experiments.append({
            'exp_name' : exp_name,
            'dataABC' : data, 
            'modelABC' : model,
        })
        
    def add_all(self, experiments:list):
        for exp in experiments:
            self.add(exp['exp_name'], exp['dataABC'], exp['modelABC'])
            # self.add(exp['exp_name'], exp['modelABC'], exp['dataABC'])  # => [Error]

    @abstractmethod
    def exec(self):
        """
        「実験」を実装する。

        【実装例】 Train
            for exp in self.experiments:
                exp_name = exp['exp_name']
                data_train = exp['data'].get_train()
                model = exp['mdoel']
                # train / eval
        """
        raise NotImplementedError()



class Basic_ExpTrain(ExperimentABC):
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



# 平均二乗誤差を計算する関数
from sklearn.metrics import mean_squared_error

class Basic_ExpEvaluate(ExperimentABC):
    """
    Evaluate の抽象化
    """
    def __init__(self, experiments:list=None, do_exec=False):
        super().__init__(experiments)
        # qini, score を保存する dict
        self.result_dicts = {
            ### イメージ
            # 'logis' : {
            #     'qini': {'train': None, 'test': None},
            #     'score': {'train': None, 'test': None}, 
            # }
        }
        self.results = {}

    def exec(self, eval_metric=['RMSE'], print_eval=True, is_output_csv=False, output_rootpath=''):
        """
        モデルを評価（Evaluate）する。
        """
        ##<<--  evaluate  -->>##
        for exp in self.experiments:
            exp_name = exp['exp_name']
            dataABC = exp['dataABC']
            model = exp['modelABC']

            result = model.predict(dataABC)
            y_df = dataABC.Y_valid  #pd.concat([dataABC.Y_train, ])
            # print(f'[Info] exp_name : {exp_name} ======================')
            # print(f'\n【result】 : \n{result}')
            # print(f"\n【y_df】 : \n{y_df.reset_index()}")

            if print_eval:
                if 'RMSE' in eval_metric:
                    print(result, y_df)
                    print('\n\n[Info] MSE  : ', mean_squared_error(y_df, result))
                    print('\n\n[Info] RMSE : ', np.sqrt(mean_squared_error(y_df, result)))

            if is_output_csv:
                result_df = pd.concat([result, y_df.reset_index()], axis=1)
                # フォルダ作成
                if output_rootpath == '':
                    output_rootpath = exp['dataABC'].dataPPP.dir_path
                out_dirpath = osp.join(output_rootpath, f'{model.name}_result')
                os.makedirs(out_dirpath, exist_ok=True)
                result_df.to_csv(osp.join(out_dirpath, f'{exp_name}.csv'))

        print("[Info] Experiment compleated!")
