import os 
import pandas as pd
from split_pictures import split_and_save_images
from select_pictures import get_random_pics
from count import detect_and_count
from extract_gpx import extract_gpx
from typing import List

def detection_pipeline(input_folder, process_folder, number, seed, model_path, conf):
    """
    ##  ※）この関数、およびライブラリは.jpgファイルのみに対応しています。
    
    
    ### 引数について
    - input_folder: 検出したい画像が入っているフォルダのパス
    - process_folder: 処理を行うときの一時フォルダ
    - number: 分割した画像のうち何枚を選択して検出するか（default=5）
    - seed: シード値。結果を固定したいときに使う（default=42）
    - model_path: インストールした5種の外来植物を検出するYOLOv8モデルのパス
    - conf: 検出時の閾値（0 <= conf <= 1）
    
    
    ---
    ### 返るpd.DataFrameの構成
    - image_path: 画像名
    - Latitude: 緯度
    - Longitude: 経度
    - DateTimeOriginal: 撮影日時
    - これにフランスギク（france）、ヒメジョオン（joon）、キクイモ（kikuimo）、オオハンゴンソウ（oohangonsou）、オオキンケイギク（ookinkeigiku）
    の5種の検出数がでます"""
    
    ### 入力チェック
    if not isinstance(number, int):
        try:
            number = int(number[0]) if isinstance(number, (list, tuple)) else int(number)
        except Exception:
            raise ValueError(f"'number' must be int, got: {type(number)} -> {number}")
    if not isinstance(seed, int):
        try:
            seed = int(seed)
        except Exception:
            raise ValueError(f"'seed' must be int, got: {type(seed)} -> {seed}")
    if not (0.0 <= float(conf) <= 1.0):
        raise ValueError(f"'conf' must be in [0,1], got: {conf}")
    
    ### 作業フォルダの準備
    if not os.path.exists(process_folder):
        os.makedirs(process_folder)
        
    ### 画像の分割
    # 分割された画像の保存先
    splited_pics_folder = os.path.join(process_folder, "splited")
    if not os.path.exists(splited_pics_folder):
        os.makedirs(splited_pics_folder)
    
    # 分割の実行
    split_and_save_images(input_folder, splited_pics_folder, target_size=640)
    print(f"finished splitting images in {input_folder}")

    ### 画像の選択
    selected_picture_paths = get_random_pics(splited_pics_folder, number=number, seed=seed)
    print(f"finished selecting pictures for detection")
    
    ### 画像から位置情報を取り出す
    df_gpx = extract_gpx(input_folder)
    print(f"finished extracting gpx data from pictures in {input_folder}")
    
    ### 検出し、画像中にある物体の数を数える
    df_count = detect_and_count(model_path=model_path, selected_picture_paths=selected_picture_paths, conf=conf)
    print(f"finished detecting pictures in {input_folder}")
    
    ### 結果のマージ
    try: 
        df_merged = pd.merge(df_gpx, df_count, on="image_path")
    except Exception as e:
        print(f"[ERROR] merge failed: {e}")
        return pd.DataFrame(columns=["image_path", "Latitude", "Longitude", "DateTimeOriginal"])
    
    print(f"finished merging dataframes")
    
    if df_merged.empty:
        print(f"[INFO] merged dataframe is empty")
    
    return df_merged