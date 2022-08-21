#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils, to_categorical
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os


# In[2]:


# 行程一覧

#0 scansnapでスキャン　数字は5桁に設定する
#1 元となるjpg読み込み　
#2 角度補正　　
#3 手書き数字認識（薬剤量）
    # 手書き数字領域の切り取り、数字毎のファイル作成
    # 数字ファイルの前処理
    # kerasを用いたMNISTの機械学習
#===================================================       
    # 数字認識の実行
    # 認識した数字をCSVに出力
#===================================================  
#4 バーコードとマークシート
    # バーコード認識
    # 固有番号の認識
    # マークシート認識
    # マークシート・バーコード・固有番号をCSVファイルに出力
#5 #3と#4のCSVファイルのマージ

#6 jpg複数枚で上記を実施


# In[4]:


# 数字認識の実行関数の定義
def recognizing(i, k):
    # recognizing関数の設定
    # kerasでdeep learningを行ったデータの保存先(自分の環境に応じて適宜変更)
    dirpath = os.path.expanduser('~/test/marksheet/keras/')

    # 画像の保存先 
    imgpath = os.path.expanduser(f"~\\eti_database\\tegakinum_drug\\num_output_{i:05d}\\")
    
    # 画像の読み込みと表示
    img = cv2.imread(imgpath + f"/tegaki_{k:02d}_prepro.jpg")
    
    # 入力画像のパラメータ
    img_width = 28 # 入力画像の幅
    img_height = 28 # 入力画像の高さ
    img_ch = 1 # 1ch画像（グレースケール）で学習

    # 入力データ数
    num_data = 1

    # 保存したモデル構造の読み込み
    model = model_from_json(open(dirpath + "model.json", 'r').read())

    # 保存した学習済みの重みを読み込み
    model.load_weights(dirpath + "weight.hdf5")

    # カメラ画像の整形
    img = cv2.imread(imgpath + f"tegaki_{k:02d}_prepro.jpg")
    
    # グレースケールに変換
    # 白黒反転,リサイズ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # グレースケール
    th = cv2.bitwise_not(gray) # 白黒反転
    th = cv2.resize(th,(img_width, img_height), cv2.INTER_CUBIC) # 訓練データと同じサイズに整形
    
    # float32に変換して正規化
    th = th.astype('float32')
    th = np.array(th)/255

    # モデルの入力次元数に合わせてリサイズ
    th = th.reshape(num_data, img_height, img_width, img_ch)

    # 分類機に入力データを与えて予測（出力：各クラスの予想確率）
    predict_y = model.predict(th)
 
    # 最も確率の高い要素番号（=予想する数字）
    predict_number = np.argmax(predict_y) 

    # 予想した数字の正答確率
    probability_number = max(np.round(predict_y[0], 3))*100
    
    # 予測結果の表示
    print("predict_y:", np.round(predict_y[0], 3))  # 出力値 小数点3以下
    print("predict_number:", predict_number)  # 予測した数字
    print("probability_number:", probability_number)
    return predict_number, probability_number

    