import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
from datetime import datetime
import glob

def run_yolo_detection(image_path):
    """YOLO物体検出を実行"""
    if not os.path.exists(image_path):
        print(f"画像ファイルが見つかりません: {image_path}")
        return
    
    # YOLOモデルを読み込み（YOLOv8n）
    print("YOLOモデルを読み込み中...")
    model = YOLO('yolov8n.pt')
    
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return
    
    print(f"画像サイズ: {image.shape}")
    print("YOLO推論を実行中...")
    
    # YOLO推論を実行
    results = model(image)
    
    # 結果を画像に描画
    annotated_image = image.copy()
    detection_count = 0
    
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
                
                # 確信度が0.5以上の場合のみ描画
                if confidence >= 0.5:
                    detection_count += 1
                    print(f"検出 {detection_count}: {class_name} (確信度: {confidence:.2f})")
                    
                    # バウンディングボックスを描画
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # ラベルテキスト（クラス名と確信度）
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # ラベルの背景サイズを計算
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    # ラベルの背景を描画
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        (0, 255, 0),
                        -1
                    )
                    
                    # ラベルテキストを描画
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )
    
    print(f"合計 {detection_count} 個のオブジェクトを検出しました。")
    
    # 出力ファイル名を生成（上書きを防ぐため連番付き）
    original_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(original_filename)
    desktop_path = r"C:\Users\filqo\OneDrive\Desktop"
    
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

def main():
    if len(sys.argv) < 2:
        print("使用方法: python yolo_detection_cli.py <画像ファイルパス>")
        print("例: python yolo_detection_cli.py sample.jpg")
        
        # デスクトップから画像ファイルを検索
        desktop_path = r"C:\Users\filqo\Box\main\研究"
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        print(f"\nデスクトップの画像ファイル:")
        found_images = []
        for ext in image_extensions:
            found_images.extend(glob.glob(os.path.join(desktop_path, ext)))
        
        if found_images:
            for i, img in enumerate(found_images[:10], 1):  # 最初の10個まで表示
                print(f"{i}. {os.path.basename(img)}")
            print(f"\n見つかった画像: {len(found_images)}個")
        else:
            print("デスクトップに画像ファイルが見つかりませんでした。")
        return
    
    image_path = sys.argv[1]
    run_yolo_detection(image_path)

if __name__ == "__main__":
    main()