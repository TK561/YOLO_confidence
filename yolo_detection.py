import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime

def select_images():
    """エクスプローラーから複数の画像ファイルを選択"""
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを非表示
    
    file_paths = filedialog.askopenfilenames(
        title="画像ファイルを選択してください（複数選択可）",
        filetypes=[
            ("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("すべてのファイル", "*.*")
        ]
    )
    
    root.destroy()
    return file_paths

def apply_mosaic(image, x1, y1, x2, y2, mosaic_size=15):
    """指定領域にモザイクを適用"""
    # 領域を切り出し
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    
    # モザイク処理
    h, w = roi.shape[:2]
    roi_small = cv2.resize(roi, (w // mosaic_size, h // mosaic_size))
    roi_mosaic = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 元の画像に適用
    image[y1:y2, x1:x2] = roi_mosaic

def detect_license_plate_area(image, car_box):
    """車のバウンディングボックス内でナンバープレートの可能性がある領域を検出"""
    x1, y1, x2, y2 = car_box
    car_height = y2 - y1
    car_width = x2 - x1
    
    # ナンバープレートは通常車の下部1/3にある
    # 前面と後面の両方を考慮
    plate_areas = []
    
    # 後部ナンバー（下部中央）
    rear_y1 = y1 + int(car_height * 0.6)
    rear_y2 = y2
    rear_x1 = x1 + int(car_width * 0.2)
    rear_x2 = x2 - int(car_width * 0.2)
    plate_areas.append((rear_x1, rear_y1, rear_x2, rear_y2))
    
    # 前部ナンバー（下部前方）
    if car_width > car_height:  # 横向きの車
        front_y1 = y1 + int(car_height * 0.6)
        front_y2 = y2
        front_x1 = x1
        front_x2 = x1 + int(car_width * 0.3)
        plate_areas.append((front_x1, front_y1, front_x2, front_y2))
    
    return plate_areas

def process_single_image(model, image_path, desktop_path):
    """単一の画像に対してYOLO物体検出を実行"""
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return False
    
    # YOLO推論を実行
    results = model(image)
    
    # 結果を画像に描画
    annotated_image = image.copy()
    
    # クラスごとの色を定義（80クラス分のカラーパレット）
    np.random.seed(42)  # 再現性のため固定シード
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # バウンディングボックスの座標
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 確信度とクラス
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # 確信度が0.7以上の場合のみ描画
                if confidence >= 0.7:
                    # クラスIDに基づいて色を選択
                    color = colors[class_id].tolist()
                    
                    # バウンディングボックスを描画
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 5)
                    
                    # ラベルテキスト（クラス名と確信度）
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # ラベルの背景サイズを計算
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                    )
                    
                    # ラベルの背景を描画
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    
                    # ラベルテキストを描画（白色で見やすく）
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3
                    )
    
    # 出力ファイル名を生成（上書きを防ぐため連番付き）
    original_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(original_filename)
    
    # タイムスタンプ付きファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{name}_yolo_detected_{timestamp}{ext}"
    output_path = os.path.join(desktop_path, output_filename)
    
    # 万が一同名ファイルが存在する場合の連番処理
    counter = 1
    while os.path.exists(output_path):
        output_filename = f"{name}_yolo_detected_{timestamp}_{counter:03d}{ext}"
        output_path = os.path.join(desktop_path, output_filename)
        counter += 1
    
    # 結果画像を保存
    cv2.imwrite(output_path, annotated_image)
    print(f"検出結果を保存しました: {output_path}")
    
    return True

def run_yolo_detection():
    """YOLO物体検出を実行（複数画像対応）"""
    # 画像ファイルを選択
    image_paths = select_images()
    
    if not image_paths:
        print("画像が選択されませんでした。")
        return
    
    print(f"{len(image_paths)}枚の画像が選択されました。")
    
    # YOLOモデルを読み込み（YOLOv8n）
    print("YOLOモデルを読み込んでいます...")
    model = YOLO('yolov8n.pt')
    
    desktop_path = r"C:\Users\filqo\OneDrive\Desktop"
    
    # 各画像を処理
    success_count = 0
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n処理中 ({i}/{len(image_paths)}): {os.path.basename(image_path)}")
        if process_single_image(model, image_path, desktop_path):
            success_count += 1
    
    print(f"\n処理完了: {success_count}/{len(image_paths)}枚の画像を正常に処理しました。")

if __name__ == "__main__":
    run_yolo_detection()