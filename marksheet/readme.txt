
# 環境
 jupyter notebookではGithubで行うバージョン管理が難しいため、jupyter labに移行した
 jupyter labの拡張機能を用いて、jupyter lab上でGithubのcommit push pullができるようにした

# ディレクトリの構成
　eti_database
    ├──left.marker.jpg
    ├──right.marker.jpg
    ├──marker.jpg
    │
    ├──image─┬─ 00001.jpg
    │        ├─ 00001_crop.jpg
    │        ├─ 00001_angle.jpg
    │        ├─ 00001_no.jpg    
    │        ├─ 00002.jpg ... 
    │
    │ 
    ├──tegakinum_drug ┬── num_output_00001 ──┬─ tegaki_01.jpg
    │                 │                      ├─ tegaki_01_prepro.jpg 
    │                 │                      ├─ tegaki_02.jpg ...
    │                 │
    │                 ├── num_output_00002
    │                 ├── ...
    │
    ├──tegakinum_time ┬── num_output_00001 ──┬─ tegaki_01.jpg
    │                 │                      ├─ tegaki_01_prepro.jpg 
    │                 │                      ├─ tegaki_02.jpg ...
    │                 │
    │                 ├── num_output_00002
    │                 ├── ...
    │
    ├──csv ┬─ 00001_drug.csv
    │      ├─ 00002_drug.csv
    │      ├─ ... 
    │      ├─ 00001_mark.csv
    │      ├─ 00002_mark.csv
    │      ├─ ... 
    │   
    ├──concatcsv ┬─ 00001.csv
    │            ├─ 00002.csv
    │
    ├──keras
    └── 

  test
    └──marksheet ┬──eti_data.ipynb
                 ├──readme.txt
                 ├──right.marker.jpg
                 ├──marker.jpg
                 └──mylib ┬──angle_numblock12.py
                          ├──barcode_mark.py   
                          ├──preprocess.py   
                          └──recognition.py
    
    
# ディレクトリの説明
 *tyamaの個人PCのGithubローカルリポジトリは"C:\Users\tyama\test\marksheet"
  ここに手書き文字認識、マーク・バーコード認識を行うコード"eti_data.ipynb"が格納されている
 *元JPG/marker file, 生成されるJPG/CSV fileは下記ディレクトリになければならない
　tyama個人PC: "C:\Users\tyama\eti_database"
  医局PC: "C:\Users\STCCC-menber\eti_database"

　上記を守らないと関数がうまく実行されない
　"eti_data.ipynb"で使用する自作関数は　\test\marksheet\mylib　に格納されている

# 行程一覧
　#0 ディレクトリを\Users\"user名"\eti_database　に移動
　#1 元となるjpg読み込み　
　#2 角度補正　　　                 　　　　　 --- angle_numblock12.pyにあるrotatefile(i)
　#3 バーコードとマークシート
    # バーコード認識
    # 固有番号の認識
    # マークシート認識
    # マークシート/バーコード/固有番号をCSV出力 --- barcode_mark.py 内のbarcodemark(i)
  #4 手書き数字認識（薬剤量）
    # 手書き数字領域の切り取り　           　　--- angle_numblock12.py 内のtrim_tegaki(i)
    # 数字毎のファイル作成                 　　--- angle_numblock12.py 内のmakenum_trim(i)
    # 数字ファイルの前処理                 　　--- preprocess.py 内のpreprocessing(k)
    # kerasを用いたMNISTの機械学習
    # 数字認識の実行　　　　　　　　　　　  　　--- preprocess.py 内のpreprocessing(k)    
    # 認識した数字とその予測確率　　　　　  　　--- recognition.py 内のrecognizing(i, k)
    # CSV出力
  #5 #3と#4のCSVファイルのマージ
  
  #6 jpg複数枚で上記を#1-5を実施
