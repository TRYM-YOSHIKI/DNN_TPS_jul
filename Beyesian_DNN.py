import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold


# データの前処理------------------------------------------------------
# csvファイルからPandas DataFrameへ読み込み
train = pd.read_csv('train.csv', delimiter=',', low_memory=False)

# trainデータを入力データとラベルに分割する
X = train.drop(['date_time'], axis=1).drop(['target_carbon_monoxide'], axis=1).drop(['target_benzene'], axis=1).drop(['target_nitrogen_oxides'], axis=1).values
Y_1 = train.target_carbon_monoxide.values  # carbon_monoxide(一酸化炭素)
Y_2 = train.target_benzene.values          # benzene(ベンゼン)
Y_3 = train.target_nitrogen_oxides.values  # nitrogen_oxides(窒素酸化物)
Y_lst = [Y_1, Y_2, Y_3]


# RMSLE カスタム評価関数 #####################
from keras import backend as K
msle = keras.metrics.MeanSquaredLogarithmicError()
def root_mean_squared_logarithmic_error(y_true, y_pred):
    return K.sqrt(msle(y_true, y_pred))


#メイン-------------------------------------------------------------
def main():
    # ベイズ最適化実行
    global y
    for y in Y_lst:
        optimizer = bayesOpt()
        print(optimizer.res)


#ベイズ最適化---------------------------------------------------------
def bayesOpt():
    # 最適化するパラメータの下限・上限
    pbounds = {
        'l1': (10, 100),
        'l2': (10, 100),
        'l1_drop': (0.0, 0.5),
        'l2_drop': (0.0, 0.5),
        'epochs': (5, 500),
        'batch_size': (64, 2048)
    }
    # 関数と最適化するパラメータを渡す
    optimizer = BayesianOptimization(f=validate, pbounds=pbounds)
    # 最適化
    optimizer.maximize(init_points=5, n_iter=10, acq='ucb')
    return optimizer


#評価------------------------------------------------------------------
def validate(l1, l2, l1_drop, l2_drop, epochs, batch_size):

    #モデルを構築&コンパイル----------------------
    def set_model():
        #モデルを構築
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(input_num,)),
            keras.layers.Dense(int(l1), activation='relu'),
            keras.layers.Dropout(l1_drop),
            keras.layers.Dense(int(l2), activation='relu'),
            keras.layers.Dropout(l2_drop),
            keras.layers.Dense(1, activation='linear')
        ])
        #モデルをコンパイル
        model.compile(optimizer='adam', 
                    loss='mean_squared_error')
        return model


    #交叉検証------------------------------------
    def Closs_validate():
        # 交差検証を実行
        valid_scores = []  # 評価を格納する配列
        kf = KFold(n_splits=5, shuffle=True, random_state=42) #データの分割の仕方を決定
        for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
            X_train, X_valid = X[train_indices], X[valid_indices]
            y_train, y_valid = y[train_indices], y[valid_indices]

            global input_num
            input_num = X_train.shape[1]

            # モデルをセット
            model = set_model()
            
            # 学習させる
            model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    verbose=0)

            # テストデータを適用する
            y_valid_pred = model.predict(X_valid)[:,0]
            
            # スコアを求める
            score = r2_score(y_valid, y_valid_pred)

            # 評価を格納する
            valid_scores.append(score)

        cv_score = np.mean(valid_scores)
        return cv_score
        
    return Closs_validate()


if __name__ == '__main__':
    main()