#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
import glob
from IPython import display


# In[2]:


# 行程一覧

#1 元となるjpg読み込み　
#2 角度補正　　
#3 手書き数字認識（薬剤量）
    # 手書き数字領域の切り取り、数字毎のファイル作成
#===================================================
    # 数字ファイルの前処理
#===================================================
    # kerasを用いたMNISTの機械学習
    # 数字認識の実行
    # 認識した数字をCSVに出力
#4 バーコードとマークシート
    # バーコード認識
    # 固有番号の認識
    # マークシート認識
    # マークシート・バーコード・固有番号をCSVファイルに出力
#5 #3と#4のCSVファイルのマージ

#6 jpg複数枚で上記を実施


# In[3]:


# 認識する元画像"i.jpg"の指定
# 認識する手書き数字画像"tegaki_k.jpg"の指定
# は後半のfor構文にあります


# In[4]:


# 処理で用いる変数の定義
new_size = (88, 88)
pad = 5


# In[5]:


# 前処理関数の定義

# img内の輪郭cntrsを背景色で埋める
def fill_unnecessary_area(img, cntrs, back_color=255):
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        img[y:(y + h), x:(x + w)] = back_color
    return img

# 膨張と収縮（モルフォロジー変換）
def morph_transformation(img):
    # ----- モルフォロジー変換 -----
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    ret_img1 = cv2.dilate(img, kernel_1, iterations=2)  # 膨張
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ret_img = cv2.erode(img, kernel_2)  # 収縮
    return ret_img

# 輪郭抽出した矩形の縦横比を変えない最大の辺の長さ(横, 縦)を返す
def get_maxrect_size(w, h, side_length):
    size = round(side_length * 0.8)
    aspect_ratio = w / h
    if aspect_ratio >= 1:
        return size, round(size / aspect_ratio)
    else:
        return round(size * aspect_ratio), size
    
    
# 抽出した矩形のパラメータ(x, y, w, h)にpad分余白を持たせる
def padding_position(x, y, w, h, pad):
    return x - pad, y - pad, w + pad * 2, h + pad * 2
    tmp_img = fill_unnecessary_area(img_inv2, contours) 
    x, y, w, h = cv2.boundingRect(max_area)
 
    if x >= pad and y >= pad:
        x, y, w, h = padding_position(x, y, w, h, pad)
 
    # ----- モルフォロジー変換 -----
    tmp_img = morph_transformation(tmp_img)
    # ----- 矩形の縦横比を保ったままリサイズする -----
    cropped = tmp_img[y:(y + h), x:(x + w)]
    new_w, new_h = get_maxrect_size(w, h, new_size[0])
    new_cropped = cv2.resize(cropped, (new_w, new_h))
    return new_cropped

# 重心を求め、中心に移動
def move_to_center(img, new_size):
    mu = cv2.moments(img)
    cx,cy= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])

    # 移動量の計算
    tx = new_size[1] / 2 - cx
    ty = new_size[0] / 2 - cy
    # x軸方向にtx, y軸方向にty平行移動させる
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dst = cv2.warpAffine(img, M, new_size)
    return dst


# In[6]:

# 前処理全体の関数の定義
def preprocessing(k):
    tegaki = f'tegaki_{k:02d}.jpg'
    
    # グレースケール変換、ブラ―、二値化
    img = cv2.imread(tegaki)
    img_gray = cv2.imread(tegaki, cv2.IMREAD_GRAYSCALE)
    img_blur1 = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_inv = cv2.threshold(img_blur1, 245, 255, cv2.THRESH_BINARY_INV)[1]

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # 外接する長方形の輪郭形成
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    output2 = cv2.rectangle(img,(x,y) ,(x+w,y+h), (0,0,255), 2)

    # img内の輪郭cntrsを背景色で埋める
    img_inv = fill_unnecessary_area(img_inv,cnt,back_color=255)
 
    # 抽出した矩形のパラメータ(x, y, w, h)にpad分余白を持たせる
        # ----- モルフォロジー変換 -----
        # ----- 矩形の縦横比を保ったままリサイズする -----
    new_cropped= padding_position(x, y, w, h, pad)

    # 重心を求め、中心に移動
    dst = move_to_center(img_inv, new_size)

    # 膨張と収縮（モルフォロジー変換）
    ret_img = morph_transformation(dst)

    # 白黒反転を戻して、ファイル出力
    img_prepro = 255 - ret_img
    cv2.imwrite(f'tegaki_{k:02d}_prepro.jpg', img_prepro)
    print(f"前処理完了_tegaki_{k:02d}_prepro")
    
# In[ ]:




