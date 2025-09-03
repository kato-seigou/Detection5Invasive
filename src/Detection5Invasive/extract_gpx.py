import os
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Union, Any, Optional, Tuple

def get_exif_data(image_path: str) -> Optional[Dict[str, Any]]:
    """画像ファイルからExifデータを取得する"""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is not None:
            exif = {}
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif[tag_name] = value
            return exif
        else:
            print(f"No Exif data found in image {image_path}")
            return None
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

def get_gps_info(exif_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """ExifデータからGPS情報を抽出する"""
    gps_info = exif_data.get('GPSInfo')
    if not gps_info:
        return None, None
    
    gps: Dict[str, Any] = {}
    for k, v in gps_info.items():
        key = ExifTags.GPSTAGS.get(k, k)
        gps[key] = v
        
    def _rat_to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            pass
        try:
            if isinstance(x, (tuple, list)) and len(x) == 2:
                num, den = x
                den = float(den)
                return float(num) / den if den != 0 else 0.0
        except Exception:
            return 0.0
        
    def _dms_to_deg(seq: Any) -> Optional[float]:
        # 度分秒（3要素）を10進法へ、要素が分数でもOK
        try:
            if not isinstance(seq, (list, tuple)) or len(seq) != 3:
                return None
            d = _rat_to_float(seq[0])
            m = _rat_to_float(seq[1])
            s = _rat_to_float(seq[2])
            return d + (m / 60.0) + (s / 3600.0)
        except Exception:
            return None
        
    lat = lon = None
    
    lat_val = gps.get("GPSLatitude")
    lat_ref = gps.get("GPSLatitudeRef")
    lon_val = gps.get("GPSLongitude")
    lon_ref = gps.get("GPSLongitudeRef")
    
    # refを安全に文字列化、大文字化へ
    def _norm_ref(v: Any) -> str:
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8", errors="ignore")
            except:
                v = str(v)
        return str(v).upper()

    try:
        if lat_val is not None and lat_ref is not None:
            lat_deg = _dms_to_deg(lat_val)
            if lat_deg is not None:
                lat = -lat_deg if _norm_ref(lat_ref) == "S" else lat_deg
        if lon_val is not None and lon_ref is not None:
            lon_deg = _dms_to_deg(lon_val)
            if lon_deg is not None:
                lon = -lon_deg if _norm_ref(lon_ref) == "W" else lon_deg
    except Exception:
        pass
    
    return lat, lon

def _parse_datetime(dt_value: Any) -> Optional[datetime]:
    """Exif の日時文字列をできるだけ頑丈にdatetimeに変換"""
    if dt_value is None:
        return None
    if isinstance(dt_value, bytes):
        try:
            dt_value = dt_value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if not isinstance(dt_value, str):
        dt_value = str(dt_value)
        
    patterns = [
        "%Y:%m:%d %H:%M:%S",  # 典型的なExif
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in patterns:
        try:
            return datetime.strptime(dt_value, fmt)
        except ValueError:
            continue
    return None

def extract_gpx(folder_path):
    """
    folder_path: 位置情報を取り出したい画像が入っているフォルダのパス
    
    返されるdfの列の内容
    image_path: ファイル名
    DateTimeOriginal: 撮影日時
    Latitude: 緯度
    Longtitude: 経度
    """
    data: List[Dict[str, Any]] = []
    
    # 拡張子フィルタ
    exts = {".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return pd.DataFrama(columns=["image_path", "DateTimeOriginal", "Latitude", "Longitude"])
    
    # フォルダ直下のみ
    image_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    
    for p in image_files:
        image_path = str(p)
        exif_data = get_exif_data(image_path)
        exif_date = None
        lat, lon = None, None
        
        if exif_data:
            # 撮影日時
            for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
                if key in exif_data:
                    parsed = _parse_datetime(exif_data.get(key))
                    if parsed:
                        exif_date = parsed
                        break
        
            # GPS情報
            lat, lon = get_gps_info(exif_data)
            
        name_norm = p.name.replace(".JPG", ".jpg")
        
        data.append({
            "image_path": name_norm,
            "DateTimeOriginal": exif_date,
            "Latitude": lat,
            "Longitude": lon
        })

    df = pd.DataFrame(data)
    return df