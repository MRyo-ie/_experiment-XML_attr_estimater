import os.path as osp
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod



class DataPPP():
    """
    (Abstruct) ML Data PreProcess Pattern.

    実験（学習 → score, qini曲線）に使用する train/test データの生成をパターン化する。
    バラバラだったので、統一しやすいように規格化した。
    
    【目標】
        1. datasetの前処理 を抽象化する。
            ・ X_train, X_val, Q_train, Q_val, A_train, At_val,  に統合・分割する。
            ・ データ構造ごとに、一般的な形でデータを整形する。
                ・ .csv → pd.DataFrame
                ・ json, xml → dict, list
                ・ 画像 → 
                ・ 文章 → 
        2. カラム名を記号に変えて、モデルが読めるようにする。
            ・ X
                ・ dict_n
                ・ num_n
                ・ date_n
                ・ img_n
            ・ Q, Y
                ・ y_n
    """
    def __init__(self):
        self.dir_path = None
        # メタ情報
        self.clmns_conv_dict = {}
        self.clmns_XQA = {
            'X' : None,
            'Q' : None,
            'Y' : None,
        }
        # all
        self.df = None
        self.X = None
        self.Q = None
        self.Y = None
        # train
        self.X_train = None
        self.Q_train = None
        self.Y_train = None
        # valid
        self.X_valid = None
        self.Q_valid = None
        self.Y_valid = None
        # test
        self.X_test = None
        self.Q_test = None
        self.Y_test = None
        # flags
        self.is_exist_Q = True
    
    def load_data(self, path: str):  # , n=None
        self.dir_path = Path(path).resolve().parents[0]
        self.df = pd.read_csv(path)            
    
    def conv_datatype_unify(self, clmns_conv_dict):
        # データの企画を統一する（主にstr, objectを変換）
        # README.md を参照。
        self.clmns_conv_dict = clmns_conv_dict
        self.df = self.df.rename(columns=self.clmns_conv_dict)

        for clmn in self.df.columns.values:
            if 'date' in clmn:
                self.df[clmn] = pd.to_datetime(self.df[clmn])
            if 'stream' in clmn:
                self.df[clmn] = pd.to_datetime(self.df[clmn])
                # print(self.df[clmn])

    def split_XQY(self, clmns_conv_dict,
                        X_clmns: list = None, 
                        Y_clmns: list = None, 
                        Q_clmns: list = None):  # , n=None
        self.conv_datatype_unify(clmns_conv_dict)
        self.clmns_XQA = {
            'X' : X_clmns, 
            'Q' : Q_clmns, 
            'Y' : Y_clmns, 
        }
        self.X = pd.DataFrame(self.df, columns=X_clmns)
        self.Q = pd.DataFrame(self.df, columns=Q_clmns)
        self.Y = pd.DataFrame(self.df, columns=Y_clmns)
        if Q_clmns is None:
            self.is_exist_Q = False

    def split_train_valid_test(self, do_valid=True, do_test=False, valid_size=0.3, test_size=0.2, random_state=30, is_shuffle=True):
        self.X_train = self.X.copy()
        self.Y_train = self.Y.copy()
        # 時系列データの場合は、分割、シャッフルはダメ。
        # 考える必要あり。
        if do_valid:
            if self.is_exist_Q:
                X_train, X_valid, Q_train, Q_valid, Y_train, Y_valid =  train_test_split(
                        self.X_train, self.Q_train, self.Y_train, 
                        test_size=valid_size, random_state=random_state, shuffle=is_shuffle)  # , stratify=self.Y_train
            else:
                X_train, X_valid, Y_train, Y_valid =  train_test_split(
                        self.X_train, self.Y_train, 
                        test_size=valid_size, random_state=random_state, shuffle=is_shuffle)  # , stratify=self.Y_train
            self.X_valid, self.Y_valid = pd.DataFrame(X_valid), pd.DataFrame(Y_valid)
            if self.is_exist_Q:
                self.Q_valid = pd.DataFrame(Q_valid)
            # print(f'\n[Info] self.Y_valid : \n{self.Y_valid}')
        if do_test:
            if self.is_exist_Q:
                X_train, X_test, Q_train, Q_test, Y_train, Y_test =  train_test_split(
                        self.X_train, self.Q_train, self.Y_train, 
                        test_size=test_size, random_state=random_state, shuffle=is_shuffle)  # , stratify=self.Y_train
            else:
                X_train, X_test, Y_train, Y_test =  train_test_split(
                        self.X_train, self.Y_train, 
                        test_size=test_size, random_state=random_state, shuffle=is_shuffle)  # , stratify=self.Y_train
            # self.X_train, self.X_test, self.Q_train, self.Q_test, self.Y_train, self.Y_test =  train_test_split(
            #         self.X.values, self.Q.values, self.Y.values, 
            #         test_size=1 / 3, random_state=30, stratify=self.Y_train
            #     )
            self.X_train, self.X_test, self.Y_train, self.Y_test = \
                pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_train), pd.DataFrame(Y_test)
            if self.is_exist_Q:
                self.Q_train, self.Q_test = pd.DataFrame(Q_train), pd.DataFrame(Q_test)




class DataABC(metaclass=ABCMeta):
    """
    (Abstruct) ML Dataset Frame.

    機械学習 Experiment（train, test）に使用する train, test データの共通規格。
    ExperimentABC, ModelABC の Input の規格(config に相当)。
    
    【目標】
        datasetの型 を抽象化する。
        用途：
            ・ train/valid/test および X/Q/Y の整理
                ・ X : features。問題文。
                ・ Q : 状態、状況、条件。質問文。（タスクごとに変化。ドメインによる差の吸収に使う）
                    ・ P(X|Q) = Y
                    例） 
                    ・ VQAのQ（質問文）
                    ・ 強化学習の状態s、Uplift modelingのtreat、
                    ・ Modelが統計手法の場合は、基本的にQは使わない。（大抵の場合、Qはドメイン特有になるため。そこも予測するのが機械学習）
                ・ Y : 
            ・ Model や 前処理プログラムの Input 規格の統一。
                ・ ModelABC を継承する、機械学習モデル のInput。
                ・ バイアス除去（IPTW, DR, SDR） のInput。
                ・ ノイズ除去 のInput。
                ・ 異常検知（？） のInput。

    【実装例】
        class Data_UpliftBasic(UpliftModelTmpl):
            def get_train(self):
                ・・・
            def get_eval(self):
                ・・・
    """
    def __init__(self, dataPPP: DataPPP=None, exist_Q=True):
        if dataPPP:
            # メタ情報
            self.clmns_conv_dict = dataPPP.clmns_conv_dict
            self.clmns_XQA = dataPPP.clmns_XQA

            # all
            self.dataPPP = dataPPP
            self.X = dataPPP.X.copy()
            self.Q = dataPPP.Q.copy()
            self.Y = dataPPP.Y.copy()

            ##  minimum  ##
            # train
            self.X_train = dataPPP.X_train.copy()
            self.Y_train = dataPPP.Y_train.copy()
            # valid
            self.X_valid = dataPPP.X_valid.copy()
            self.Y_valid = dataPPP.Y_valid.copy()
            # test
            self.X_test = dataPPP.X_test.copy()
            if type(dataPPP.Y_test) == pd.DataFrame:
                self.Y_test = dataPPP.Y_test.copy()
            else:
                self.Y_test = dataPPP.Y_test

            ##  Maximum  ##
            self.Q_train = None
            self.Q_test = None
            self.Q_valid = None
            if exist_Q:
                # Q
                self.Q_train = dataPPP.Q_train.copy()
                self.Q_valid = dataPPP.Q_valid.copy()
                self.Q_test = dataPPP.Q_test.copy()

            # data の型など、問題がないかチェックする。（未）
            # self._check()

    @abstractmethod
    def get_train(self):  #=> dict or stream
        """
        例）
            return {
                'train' : {
                    'X': self.X_train, 
                    'Q': self.Q_train, 
                    'Y': self.Y_train,
                },
                'valid' : { 
                    'X': self.X_valid, 
                    'Q': self.Q_valid, 
                    'Y': self.Y_valid,
                }
            }
        """
        raise NotImplementedError()

    @abstractmethod
    def get_eval(self):  #=> dict or stream
        """
        例）
            return {
                'train' : {
                    'X': self.X_train, 
                    'Q': self.Q_train, 
                    'Y': self.Y_train,
                },
                'valid' : { 
                    'X': self.X_valid, 
                    'Q': self.Q_valid, 
                    'Y': self.Y_valid,
                },
                'test' : { 
                    'X': self.X_test, 
                    'Q': self.Q_test, 
                    'Y': self.Y_test,
                }
            }
        """
        
    
    def _check(self, data):
        raise NotImplementedError()





