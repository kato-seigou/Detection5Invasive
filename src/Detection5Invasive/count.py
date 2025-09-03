import os
import re
import pandas as pd
from ultralytics import YOLO
from typing import List, Union, Iterable

# ファイル名を元のものに統合させる関数
def convert_filename(filename: str) -> str:
    parts = filename.split("_")
    new_filename = parts[0] + ".jpg"
    return new_filename

def _validate_inputs(selected_pictures_paths: Iterable[str]) -> List[str]:
    paths = []
    for p in selected_pictures_paths:
        if isinstance(p, str) and p.lower().endswith(".jpg") and os.path.exists(p):
            paths.append(p)
    if not paths:
        raise ValueError("no valid full path in selected_pictures_paths")
    return paths

# 検出結果をファイルごとに表示したテーブルを出力する関数
def detect_and_count(model_path, selected_picture_paths, conf):
    """
    ### 引数
    - model: 使用するモデルのパス（YOLOベース）
    - selected_pictures_paths: 検出用に選択された画像のフルパスのリスト
    - conf: 検出の閾値（0 < conf < 1）
    
    ### 返り値
    - 検出結果を格納したpd.DataFrame
    """
    # モデル読み込み
    try: 
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] failed to load model: {model_path}\n{e}")
        return pd.DataFrame(columns=["image_path"])
    
    try:
        image_paths = _validate_inputs(selected_pictures_paths=selected_picture_paths)
    except Exception as e:
        print(f"[ERROR] invalid selected_pictures_paths: {e}")
        return pd.DataFrame(columns=["image_paths"])
    
    all_result_arrays = []
    error_files = []
    processed = 0
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"[WARN] file dissappered: {image_path}")
            continue
        
        try:
            results = model.predict(
                image_path, 
                show=False,
                save=False,
                verbose=False,
                conf=conf
            )
            
            for result in results:
                # names を安全に正規化してクラス数を算出
                nm = result.names
                if isinstance(nm, dict):
                    # キーが文字列のこともあるので int に寄せる
                    try:
                        keys = [int(k) for k in nm.keys()]
                    except Exception:
                        keys = list(range(len(nm)))
                        nm = {i: list(nm.values())[i] for i in keys}
                    names_map = {int(k): v for k, v in nm.items()}
                    num_classes = (max(keys) + 1) if keys else 0
                else:
                    # list/tuple/その他iterable
                    seq = list(nm)
                    names_map = {i: n for i, n in enumerate(seq)}
                    num_classes = len(seq)

                # 検出数を0で初期化
                clscount = [0] * num_classes
                
                # boxesが空でない場合のみカウント
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            cls_id = int(box.cls[0].item())
                            if 0 <= cls_id < num_classes:
                                clscount[cls_id] += 1
                        except Exception:
                            continue
                
                # 行を組み立て
                result_array = [("image_path", os.path.basename(result.path))]
                for i in range(num_classes):
                    cname = names_map.get(i, f"cls_{i}")
                    result_array.append((cname, clscount[i]))
                all_result_arrays.append(result_array)
                
            processed += 1
            
        except Exception as e:
            print(f"[ERROR] inference failed for {image_path}: {e}")
            error_files.append(image_path)
            continue
        
    # df化
    result_dict = [dict(r) for r in all_result_arrays]
    if not result_dict:
        print(f"[INFO] no detections or all failed. processed={processed}, errors={len(error_files)}")
        return pd.DataFrame(columns=["image_path"])
    
    df = pd.DataFrame(result_dict)
    
    # 失敗ファイルのログ
    if error_files:
        print(f"[WARN] errors occured on these files: ")
        for f in error_files:
            print(" ", f)
    
    # 元ファイルへ統合
    if "image_path" not in df.columns:
        return pd.DataFrame(columns=["image_path"])
    
    df["image_path"] = df["image_path"].apply(convert_filename)
    
    # 同じファイル名で集計
    value_cols = [c for c in df.columns if c != "image_path"]
    if not value_cols:
        return pd.DataFrame(columns=["image_path"])
    df = df.groupby("image_path", as_index=False)[value_cols].sum()
    
    # サマリ
    print(f"[INFO] processed={processed}, rows={len(df)}, errors={len(error_files)}")
    
    return df