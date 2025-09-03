import cv2
import os
from pathlib import Path
from typing import List, Tuple, Union

def compute_grid(img_height: int, img_width: int, target_size: int=640) -> Tuple[int, int]:
    """
    画像サイズから、分割後のピースがtarget_sizeに近くなるように
    rows（縦分割数）、cols（横分割数）を自動計算する
    """
    rows = max(1, round(img_height / target_size))
    cols = max(1, round(img_width / target_size))
    return rows, cols



def split_image(image_path: Union[str, Path], output_folder: Union[str, Path], target_size: int=640) :
    """
    画像を分割して保存する関数
    
    image_path: 入力画像
    output_folder: 出力フォルダ
    target_size: 分割後のサイズの目標値（default=640）
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    img_height, img_width, _ = img.shape
    rows, cols = compute_grid(img_height, img_width, target_size)
    
    section_width = max(1, img_width // cols) 
    section_height = max(1, img_height // rows)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_paths = []
    for y in range(rows):
        for x in range(cols):
            start_x = x * section_width
            start_y = y * section_height
            end_x = (x + 1) * section_width if x < cols - 1 else img_width
            end_y = (y + 1) * section_height if y < rows - 1 else img_height
            
            section = img[start_y: end_y, start_x: end_x]
            
            image_filename = os.path.splitext(os.path.basename(image_path))[0]
            section_filename = f"{image_filename}_{y * cols + x + 1}.jpg"
            
            output_path = os.path.join(output_folder, section_filename)
            cv2.imwrite(output_path, section)
            output_paths.append(output_path)
    

def split_and_save_images(input_folder: Union[str, Path], output_folder: Union[str, Path], target_size: int=640) -> None:
    """
    フォルダに入っている画像を分割して任意のフォルダに格納する関数
    """
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg"))]
    
    if not image_paths:
        print("No .jpg images found in the input folder. This function handles '.jpg' or '.JPG' file only.")
        return 
        
    for image_path in image_paths:
        split_image(image_path, output_folder, target_size) 
    
    print("Images are saved and splited successfully.")