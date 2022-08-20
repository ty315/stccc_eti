#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pyocr
import pyocr.builders
import sys
from pyzbar.pyzbar import decode
import glob
import pandas as pd


# In[2]:


# 行程一覧

#1 元となるjpg読み込み　
#2 角度補正　　
#3 手書き数字認識（薬剤量）
    # 手書き数字領域の切り取り、数字毎のファイル作成
    # 数字ファイルの前処理
    # kerasを用いたMNISTの機械学習
    # 数字認識の実行
    # 認識した数字をCSVに出力
#===================================================
#4 バーコードとマークシート
    # バーコード認識
    # 固有番号の認識
    # マークシート認識
    # マークシート・バーコード・固有番号をCSVファイルに出力
#===================================================
#5 #3と#4のCSVファイルのマージ

#6 jpg複数枚で上記を実施


# In[3]:


# 課題
# わずかな点でも認識される問題
# "[]"の消去

# 角度補正は#2ですでに行った前提

# 元画像のファイル番号を取得
# あるいは、for構文で指定


# In[4]:


def barcodemark(i):
    scan = f'{i:05d}_angle.jpg'
    img = cv2.imread(scan)
    # バーコードリーダー errorは"error"で返す
    # 切り出し範囲＝img[縦方向（上）：縦方向（下）, 横方向（左）：横方向（右）]
    img_crop = img[250:330, 1500:2100]
    plt.imshow(img_crop, cmap = "gray")
    plt.show()
    im_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(img_crop, 160, 255, cv2.THRESH_BINARY)
    cv2.imwrite('barcode_g.jpg', dst)
    data = decode(Image.open('barcode_g.jpg'))
    try:
        a = data[0][0].decode('utf-8', 'ignore')
    except IndexError:
        a = 'error'
    b = a.strip('AA') 
    c = b.lstrip('0')
    print(c)
    os.remove('barcode_g.jpg')

    # 書類の固有番号の読み取り
    # 切り出し範囲＝img[縦方向（上）：縦方向（下）, 横方向（左）：横方向（右）]
    img_no = img[0:500, 2100:2400]
    marker=cv2.imread('marker.jpg', 0)
    img_gray = cv2.cvtColor(img_no, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img_gray, marker, cv2.TM_CCOEFF_NORMED)
    # 検出結果から領域の位置を取得. 類似度が最大のものを使用
    _, _, _, max_loc = cv2.minMaxLoc(result)
    # マーカーの位置から一定距離で再度トリミング
    img_no_no = img_no[max_loc[1]-300:max_loc[1]-220, max_loc[0]-120:300]
    # 前処理（グレースケール、白黒反転、二値化、反転戻し）
    img_no_no = cv2.cvtColor(img_no_no, cv2.COLOR_BGR2GRAY)
    img_no_no = cv2.bitwise_not(img_no_no)
    ret, img_no_no = cv2.threshold(img_no_no, 200, 255, cv2.THRESH_BINARY) 
    img_no_no = 255 - img_no_no
    # OCRするファイル(i_ocr.jpg)を書き出し、PILで読み込む
    cv2.imwrite(f'{i:05d}_no.jpg',img_no_no, [cv2.IMWRITE_JPEG_QUALITY, 100])
    img_ocr = Image.open(f'{i:05d}_no.jpg')
    # OCRの実行
    path='C:\\Program Files\\Tesseract-OCR\\'
    os.environ['PATH'] = os.environ['PATH'] + path
    pyocr.tesseract.TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tools = pyocr.get_available_tools()
    tool = tools[0]
    builder = pyocr.builders.TextBuilder(tesseract_layout=6)
    no = tool.image_to_string(img_ocr, lang="eng", builder=builder)
    no = int(no)
    no = f"{no:05d}"
    print("認識された固有番号:", no)

    try:
        ID = int(c)
    except ValueError:
        ID = 'error'
    fileno = f"{i:05d}"
    df1 = pd.DataFrame(
    data =[[fileno, no, ID]], 
    columns = ['fileno', 'no', 'ID']
    )
       
    # マークシートの認識
    marker=cv2.imread('marker.jpg')
    marker=cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
    w, h = marker.shape[::-1]
    img = cv2.imread(scan)
    # 画像を右半分のトリミングして、マーカーのtemplatematching
    img = img[0:3500, 1235:2430]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img, marker, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    mark_area={}
    mark_area['top_x']= min(loc[1])
    mark_area['top_y']= min(loc[0])
    mark_area['bottom_x']= max(loc[1])
    mark_area['bottom_y']= max(loc[0])
    img = img[mark_area['top_y']:mark_area['bottom_y'],mark_area['top_x']:mark_area['bottom_x']]
    cv2.imwrite('res.png',img)

    n_col = 10 # 1行あたりのマークの数
    n_row = 21 # マークの行数
    margin_top = 1 # 上余白行数
    margin_bottom = 1 # 下余白行数
    n_row = n_row + margin_top + margin_bottom # 行数 (マーク行 7行 + 上余白 3行 + 下余白 1行)
    img = cv2.resize(img, (n_col*100, n_row*100))
    img = cv2.GaussianBlur(img,(5,5),0) #ブラーをかける
    res, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = 255 - img

    result = []
    for row in range(margin_top, n_row - margin_bottom):
        tmp_img = img [row*100:(row+1)*100,]
        area_sum = [] # 合計値を入れる配列
        for col in range(n_col):
            area_sum.append(np.sum(tmp_img[:,col*100:(col+1)*100]))
        result.append(area_sum > np.max(area_sum)*0.3)

    # np.max(area_sum)は各列で160000～240000
    # 閾値はnp.maxの3割に設定（全体の約3%）

    out = []
    for x in range(len(result)):
        res = np.where(result[x]==True)[0]
        if len(res)>=1: out.append(str(res))
        else: out.append("NaN")

    df2 = pd.DataFrame(
    data = [out], 
    columns = ['loc', 'drug', 'NaN1', 'NaN2', 'NaN3', 'NaN4', 'dur_1', 'dur_2', 'dur_3', 'spo2_1', 'spo2_2', 'att', 'NaN5', 'ind', 'dam', 'mask', 'mlemon', 'igel', 'comp', 'NaN6', 'empty']
    )  
    df3 = pd.concat([df1, df2], axis=1)
    display(df3)
    os.remove('no.jpg')
    os.remove('res.png')
    os.chdir('/Users/STCCC-menber/Desktop/marksheet/markcsv')
    df3.to_csv(f'mark_{i:05d}.csv')

