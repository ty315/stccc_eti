#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
import sys
import glob
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import math


# In[2]:


# 行程一覧

#0 scansnapでスキャン　数字は5桁に設定する
#===================================================
#1 元となるjpg読み込み　
#2 角度補正　　
#3 手書き数字認識（薬剤量）
    # 手書き数字領域の切り取り、数字毎のファイル作成
#===================================================
    # 数字ファイルの前処理
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


# In[4]:


# 角度補正関数の定義
def detect_marker(file, marker):
    # 画像ファイルとテンプレートをグレースケールで読み込む
    img = cv2.imread(file, 0)
    template = cv2.imread(marker, 0)
    # テンプレートマッチング
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    # 検出結果から領域の位置を取得. 類似度が最大のものを使用
    _, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc

# 補正する回転角度(degree)を計算
def calc_rotation_angle(left_pos, right_pos):
    x = right_pos[0] - left_pos[0]
    y = right_pos[1] - left_pos[1]
    return math.degrees(math.atan2(y, x))

# 回転の実施
def rotate_img(img, angle):
    # 画像サイズ(横, 縦)から中心座標を求める
    size = tuple([img.shape[1], img.shape[0]])
    center = tuple([size[0] // 2, size[1] // 2])
    # 回転の変換行列を求める(画像の中心, 回転角度, 拡大率)
    mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    # アフィン変換(画像, 変換行列, 出力サイズ, 補完アルゴリズム)
    rot_img = cv2.warpAffine(img, mat, size, flags=cv2.INTER_CUBIC)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return rot_img


# In[6]:


# 角度補正の実行
def rotatefile(i):
    scan = f"./image/{i:05d}.jpg"
    img = cv2.imread(scan, 0)
    left_pos = detect_marker(scan, "left_marker.jpg")
    right_pos = detect_marker(scan, "right_marker.jpg")
    angle = calc_rotation_angle(left_pos, right_pos)
    rot_img = rotate_img(img,angle)
    cv2.imwrite(f'./image/{i:05d}_angle.jpg',rot_img)
    print("angle.jpg was created")

# In[7]:

# 手書き数字領域の切り取り、数字毎のファイル作成
def trim_tegaki(i):
    anglefile = f'./image/{i:05d}_angle.jpg'
    img = Image.open(anglefile)
    img_crop = img.crop((1,500, 1800, 3200))
    img_crop.save(f'./image/{i:05d}_crop.jpg', quality=95)
    img_tegaki = cv2.imread(f'./image/{i:05d}_crop.jpg')
    marker=cv2.imread('marker.jpg')
    marker=cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
    img_tegaki = cv2.cvtColor(img_tegaki, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img_tegaki, marker, cv2.TM_CCOEFF_NORMED)

    threshold = 0.9   # 類似度の設定(0~1)
    loc = np.array(np.where(result >= threshold))

    mark_area={}
    mark_area['top_x']= min(loc[1])
    mark_area['top_y']= min(loc[0])
    mark_area['bottom_x']= max(loc[1])
    mark_area['bottom_y']= max(loc[0])
    img_tegaki = img_tegaki[mark_area['top_y']:mark_area['bottom_y'],mark_area['top_x']:mark_area['bottom_x']]

    cv2.imwrite(f'./image/{i:05d}_crop.jpg',img_tegaki)
    print("crop.jpg was created")


# In[8]:

# 手書き領域の分割と、数字ファイルの保存
# 数字ファイルのトリミングと不要なファイルの除去

def makenum_trim(i):
    rows = 4  # 行数
    cols = 3  # 列数
    img_tegaki = cv2.imread(f"./image/{i:05d}_crop.jpg") 
    chunks = []
    for row_img_tegaki in np.array_split(img_tegaki, rows, axis=0):
        for chunk in np.array_split(row_img_tegaki, cols, axis=1):
            chunks.append(chunk)
    os.chdir("tegakinum_drug")
    output_dir = Path(f"num_output_{i:05d}")
    output_dir.mkdir(exist_ok=True)
    for k, chunk in enumerate(chunks):
        save_path = output_dir / f"chunk_{k:02d}.png"
        cv2.imwrite(str(save_path), chunk)

    os.chdir(f'num_output_{i:05d}')
    tegakimoji = glob.glob("*.png")

    for k in range(0, 12):
        tegaki = f"chunk_{k:02d}.png"
        img = Image.open(tegaki)
        img_tegaki = img.crop((7,10, 83, 119))
        save_path = f"tegaki_{k:02d}.jpg"
        img_tegaki.save(str(save_path), quality=95)
  
    for k in range(0,12):
        os.remove(f"chunk_{k:02d}.png")

    os.remove("tegaki_00.jpg")
    os.remove("tegaki_03.jpg")
    os.remove("tegaki_06.jpg")
    os.remove("tegaki_09.jpg")
    os.chdir("../../")
    print("tegaki.jpg was created")

# In[ ]:




