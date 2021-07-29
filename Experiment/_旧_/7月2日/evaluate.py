
from .abc_experiment import ExperimentABC
import tensorflow as tf


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




class IMDB_TextClassify_ExpEvaluate(ExperimentABC):
    """
    Evaluate の抽象化
    """
    def __init__(self, experiments:list=None, do_exec=False):
        super().__init__(experiments)

        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = 32

        self.test_ds = tf.keras.preprocessing.text_dataset_from_directory(
                    'aclImdb/test',
                    batch_size=batch_size)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)


    def exec(self):
        ##<<--  evaluate  -->>##
        for exp in self.experiments:
            exp_name = exp['exp_name']
            dataABC = exp['dataABC']
            model = exp['modelABC']
            # BERT
            loss, accuracy = model.evaluate(self.test_ds)

            print(f'Loss: {loss}')
            print(f'Accuracy: {accuracy}')

