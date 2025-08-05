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

def calculate_iou(box1, box2):
    """2つのボックス間のIoU（Intersection over Union）を計算"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 交差領域の座標
    intersect_xmin = max(x1_min, x2_min)
    intersect_ymin = max(y1_min, y2_min)
    intersect_xmax = min(x1_max, x2_max)
    intersect_ymax = min(y1_max, y2_max)
    
    # 交差領域の面積
    if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
        return 0.0
    
    intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
    
    # 各ボックスの面積
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # IoU
    union_area = box1_area + box2_area - intersect_area
    return intersect_area / union_area if union_area > 0 else 0.0

def process_single_image(model, image_path, desktop_path):
    """単一の画像に対してYOLO物体検出を実行"""
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return False
    
    # YOLO推論を実行（NMSパラメータを明示的に設定）
    results = model(image, conf=0.25, iou=0.45)
    
    # 結果を画像に描画
    annotated_image = image.copy()
    
    # クラスごとの色を定義（80クラス分のカラーパレット）
    np.random.seed(42)  # 再現性のため固定シード
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
    # デバッグ用：検出された物体の数を記録
    detection_count = 0
    
    # 検出結果を格納するリスト
    detections = []
    
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
                
                # デバッグ：すべての検出結果の詳細を表示
                print(f"  検出: {class_name}, 確信度: {confidence:.3f}, 位置: ({x1},{y1})-({x2},{y2})")
                
                # 確信度が0.25以上の場合のみ処理
                if confidence >= 0.25:
                    # 検出結果をリストに追加
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })
    
    # 重複する検出結果をフィルタリング（IoUが高い場合は確信度の高い方を残す）
    filtered_detections = []
    for i, det1 in enumerate(detections):
        is_duplicate = False
        for j, det2 in enumerate(detections):
            if i != j:
                iou = calculate_iou(det1['box'], det2['box'])
                # IoUが0.8以上の場合は重複とみなす
                if iou > 0.8:
                    # 確信度が低い方を重複として扱う
                    if det1['confidence'] < det2['confidence']:
                        is_duplicate = True
                        break
        if not is_duplicate:
            filtered_detections.append(det1)
    
    print(f"  -> フィルタリング前: {len(detections)}個, フィルタリング後: {len(filtered_detections)}個")
    
    # フィルタリング後の検出結果を描画
    for det in filtered_detections:
        detection_count += 1
        x1, y1, x2, y2 = det['box']
        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']
        
        # クラスIDに基づいて色を選択
        color = colors[class_id].tolist()
        
        # バウンディングボックスを描画
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 12)
        print(f"  描画: {class_name} at ({x1},{y1})-({x2},{y2}) with color {color}")
        
        # ラベルテキスト（クラス名と確信度）
        label = f"{class_name}: {confidence:.2f}"
        
        # ラベルの背景サイズを計算
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 6
        )
        
        # ラベルの背景を描画
        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # ラベルテキストを描画（白色で見やすく）
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.5,
            (255, 255, 255),
            6
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
    
    # デバッグ：検出された物体の数を表示
    print(f"  -> 70%以上の確信度で検出された物体: {detection_count}個")
    
    # 画像が正しく変更されているか確認
    if detection_count > 0:
        # 元画像と注釈付き画像の差分を確認
        diff = cv2.absdiff(image, annotated_image)
        diff_sum = np.sum(diff)
        print(f"  -> 画像の差分合計: {diff_sum}")
    
    # 結果画像を保存
    success = cv2.imwrite(output_path, annotated_image)
    if success:
        print(f"検出結果を保存しました: {output_path}")
    else:
        print(f"画像の保存に失敗しました: {output_path}")
    
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