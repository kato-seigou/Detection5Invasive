import os
import shutil
import random
import re
import cv2
import numpy as np
import random

from pathlib import Path
from typing import List, Dict, DefaultDict, Union
from collections import defaultdict

# フォルダから画像のパスを出力する関数の定義
def get_image_paths(folder_path: Union[str, Path]) -> List[str]:
    supported_extensions = ("jpg", "JPG")
    folder_path = str(folder_path)
    
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(supported_extensions)
    ]
    
    return image_paths

# フォルダをパスごとにサブリスト化する関数の定義
def group_files_by_id(file_list: List[str]) -> List[List[str]]:
    grouped_files: DefaultDict[str, List[str]] = defaultdict(list)
    
    for file in file_list:
        dirname = os.path.dirname(file)
        basename = os.path.basename(file)
        match = re.match(r"([a-zA-Z0-9]+)_\d+\.jpg", basename)
        if match:
            file_id = match.group(1)
            grouped_files[file_id].append(os.path.join(dirname, basename))
    
    return list(grouped_files.values())

# 特定の範囲のpixel値を持つ画像のパスを格納するリストの初期化
def extract_images(source_paths: List[str]) -> List[str]:
    output_list: List[str] = [] # 特定の範囲のpixel値を持つ画像のパスを格納するリストの初期化
    for source_path in source_paths:
        img = cv2.imread(source_path)
        if img is None:
            continue # 継続処理
        
        # OpenCVはBGRなので、HSVに変換する
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow) # この範囲内にあるピクセルのみを抽出した画像を返す
        non_zero_count_yellow = np.count_nonzero(mask_yellow) # mask画像でピクセル値が0出ないもののが図を返す
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white) # この範囲内にあるピクセルのみを抽出した画像を返す
        non_zero_count_white = np.count_nonzero(mask_white) 
        
        if non_zero_count_white == 0 and non_zero_count_yellow == 0:
            continue # 範囲内でない場合は棄却する
        
        output_list.append(source_path)
    
    return output_list

# サブリスト化されたリストからサブリスト内から任意の数の画像をランダムに取得する関数の定義
# リストを返す
def get_random_pics(folder_path: Union[str, Path], number: int, seed: int = 42) -> List[str]:
    """
    ### 引数
    - folder_path: 分割された画像が格納されているフォルダ（1つ）
    - number: 何枚の画像を検出用に選択し保存するか
    - seed: ランダムシード
    - ※ .jpgのみ対象
    
    ### 返り値
    選択された画像のフルパスのリスト
    """
    if seed is not None:
        random.seed(seed)
    
    file_list = get_image_paths(folder_path)
    grouped_file_list = group_files_by_id(file_list)
    
    # numberごとにランダムに画像を選択して保存する
    selected_list: List[List[str]] = []
    for sublist in grouped_file_list:
        sublist = extract_images(sublist)
        if not sublist:
            continue
        
        k = min(number, len(sublist))
        selected_list.append(list(random.sample(sublist, k)))
        
    # フラット化して返す
    selected_paths = [p for group in selected_list for p in group]
    return selected_paths