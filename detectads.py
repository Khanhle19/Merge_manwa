import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import sys
import json
import torch
from ultralytics import YOLO

# === Cấu hình ===
input_folder = r"f:\Manhwa_result\zfinish\hunter-academys-battle-god-961\result"  # thư mục chứa ảnh cần xử lý
output_dir = os.path.join(input_folder, "ad_detection_results")  # thư mục lưu kết quả
patches_dir = os.path.join(output_dir, "patches")  # thư mục lưu các patch

# Thay thế đường dẫn này bằng đường dẫn đến mô hình YOLOv8 của bạn
model_path = "E:/yolo/runs/detect_ads_model/weights/best.pt"  # Đường dẫn đến file model tốt nhất của bạn

# Định dạng ảnh được hỗ trợ
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Cấu hình cửa sổ trượt
patch_width = 640   # chiều rộng mỗi patch
patch_height = 640   # chiều cao mỗi patch
overlap_x = 200      # overlap theo chiều ngang
overlap_y = 200      # overlap theo chiều dọc
confidence_threshold = 0.6  # ngưỡng tin cậy tối thiểu
nms_iou_threshold = 0.5     # ngưỡng IOU cho NMS

# Cấu hình lưu patches
save_only_patches_with_detections = True  # Chỉ lưu patches có phát hiện
patch_format = "png"  # Định dạng lưu patches (png hoặc jpg)

# === Tạo thư mục lưu kết quả ===
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(patches_dir):
    os.makedirs(patches_dir)

# === Load mô hình YOLOv8 ===
def load_model():
    """Load mô hình YOLOv8 đã huấn luyện"""
    try:
        model = YOLO(model_path)
        print(f"Đã tải mô hình YOLOv8 từ {model_path}")
        # Kiểm tra thiết bị tính toán
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {device}")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLOv8: {str(e)}")
        sys.exit(1)

# === Hàm tính IoU (Intersection over Union) ===
def calculate_iou(box1, box2):
    """Tính toán IoU giữa hai khung hình"""
    # box format: [x, y, width, height]
    x1_1, y1_1 = box1['x'] - box1['width']/2, box1['y'] - box1['height']/2
    x2_1, y2_1 = box1['x'] + box1['width']/2, box1['y'] + box1['height']/2
    
    x1_2, y1_2 = box2['x'] - box2['width']/2, box2['y'] - box2['height']/2
    x2_2, y2_2 = box2['x'] + box2['width']/2, box2['y'] + box2['height']/2
    
    # Tính diện tích giao nhau
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Tính diện tích của mỗi box
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    
    # Tính IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# === Hàm NMS (Non-Maximum Suppression) ===
def non_max_suppression(boxes, iou_threshold):
    """Lọc bỏ các khung trùng lặp bằng Non-Maximum Suppression"""
    if not boxes:
        return []
    
    # Sắp xếp theo độ tin cậy giảm dần
    sorted_boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    
    selected_boxes = []
    
    while sorted_boxes:
        current_box = sorted_boxes.pop(0)
        selected_boxes.append(current_box)
        
        # So sánh với các box còn lại
        i = 0
        while i < len(sorted_boxes):
            if calculate_iou(current_box, sorted_boxes[i]) > iou_threshold:
                sorted_boxes.pop(i)
            else:
                i += 1
    
    return selected_boxes

# === Hàm tìm tất cả files ảnh trong thư mục ===
def find_all_images(directory):
    """Tìm tất cả các file ảnh trong thư mục"""
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                image_files.append(os.path.join(root, file))
    
    return image_files

# === Hàm vẽ các box lên patch ===
def draw_boxes_on_patch(patch, detections, local_x=0, local_y=0):
    """Vẽ các box phát hiện lên patch"""
    draw = ImageDraw.Draw(patch)
    
    # Cố gắng tải font, nếu không có thì dùng default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Vẽ các box phát hiện được
    for det in detections:
        # Tính tọa độ tương đối trong patch
        x = det['x'] - local_x
        y = det['y'] - local_y
        w = det['width']
        h = det['height']
        confidence = det['confidence']
        class_name = det.get('class', 'ad')
        
        # Tính tọa độ box
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Chọn màu dựa trên confidence
        color = (255, int(255 * (1 - confidence)), 0)
        
        # Vẽ box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Vẽ nhãn
        label = f"{class_name}: {confidence:.2f}"
        label_size = draw.textbbox((0, 0), label, font=font)[2:4]
        
        # Vẽ background cho text
        draw.rectangle(
            [x1, y1 - label_size[1] - 4, x1 + label_size[0] + 4, y1],
            fill=color
        )
        draw.text((x1 + 2, y1 - label_size[1] - 2), label, fill=(255, 255, 255), font=font)
    
    return patch

# === Hàm chuyển đổi kết quả YOLOv8 thành định dạng cần thiết ===
def convert_yolo_results_to_detection(results, patch_offset_x=0, patch_offset_y=0, patch_id=None, patch_info=None):
    """Chuyển đổi kết quả từ YOLOv8 sang định dạng phù hợp với code hiện tại"""
    detections = []
    
    # Xử lý các box được phát hiện
    for box in results.boxes:
        # Lấy tọa độ YOLO (x1, y1, x2, y2)
        xyxy = box.xyxy[0].cpu().numpy()  # lấy tọa độ
        
        # Chuyển sang định dạng trung tâm + kích thước
        x_center = (xyxy[0] + xyxy[2]) / 2 + patch_offset_x  # x trung tâm + offset
        y_center = (xyxy[1] + xyxy[3]) / 2 + patch_offset_y  # y trung tâm + offset
        width = xyxy[2] - xyxy[0]  # chiều rộng
        height = xyxy[3] - xyxy[1]  # chiều cao
        
        # Lấy độ tin cậy
        confidence = float(box.conf[0])
        
        # Lấy tên class
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        
        # Tạo object phát hiện
        detection = {
            'x': float(x_center),
            'y': float(y_center),
            'width': float(width),
            'height': float(height),
            'confidence': float(confidence),
            'class': class_name,
            'patch_id': patch_id,
            'patch_info': patch_info
        }
        
        detections.append(detection)
    
    return detections

# === Hàm xử lý ảnh với cửa sổ trượt 2 chiều và lưu từng patch ===
def sliding_window_detect(image_path, model):
    """Phát hiện đối tượng trong ảnh bằng cách trượt cửa sổ và lưu từng patch riêng lẻ"""
    try:
        image = Image.open(image_path)
        width, height = image.size
        print(f"Xử lý ảnh kích thước {width}x{height} pixels")
        
        # Tạo thư mục để lưu patches cho ảnh này
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_patches_dir = os.path.join(patches_dir, image_name)
        os.makedirs(image_patches_dir, exist_ok=True)
        
        all_results = []
        patch_info = []
        
        # Tính toán số lượng patch
        x_steps = max(1, int(np.ceil((width - overlap_x) / (patch_width - overlap_x))))
        y_steps = max(1, int(np.ceil((height - overlap_y) / (patch_height - overlap_y))))
        total_patches = x_steps * y_steps
        
        # Khởi tạo thanh tiến trình
        progress_bar = tqdm(total=total_patches, desc="Xử lý patches")
        
        # Duyệt qua các patch
        for y_idx in range(y_steps):
            y = min(y_idx * (patch_height - overlap_y), height - patch_height)
            y = max(0, y)  # Đảm bảo y không âm
            
            for x_idx in range(x_steps):
                x = min(x_idx * (patch_width - overlap_x), width - patch_width)
                x = max(0, x)  # Đảm bảo x không âm
                
                # Cắt patch từ ảnh gốc
                right = min(x + patch_width, width)
                bottom = min(y + patch_height, height)
                patch = image.crop((x, y, right, bottom))
                
                # ID của patch hiện tại
                patch_id = f"{y_idx:03d}_{x_idx:03d}"
                patch_filename = f"patch_{patch_id}.{patch_format}"
                patch_path = os.path.join(image_patches_dir, patch_filename)
                
                # Lưu patch tạm thời
                temp_patch_path = os.path.join(output_dir, "temp_patch.jpg")
                patch.save(temp_patch_path)
                
                # Phát hiện đối tượng với YOLOv8
                try:
                    # Dự đoán với mô hình YOLOv8
                    results = model.predict(
                        source=temp_patch_path,
                        conf=confidence_threshold,
                        verbose=False
                    )[0]  # Lấy kết quả đầu tiên
                    
                    # Chuyển đổi kết quả thành định dạng phù hợp
                    patch_detections = convert_yolo_results_to_detection(
                        results,
                        patch_offset_x=x,  # offset x
                        patch_offset_y=y,  # offset y
                        patch_id=patch_id,
                        patch_info={"x": x, "y": y, "width": right-x, "height": bottom-y}
                    )
                    
                    # Thêm kết quả vào danh sách
                    all_results.extend(patch_detections)
                    
                    # Lưu thông tin patch
                    current_patch_info = {
                        "patch_id": patch_id,
                        "x": x,
                        "y": y,
                        "width": right - x,
                        "height": bottom - y,
                        "path": patch_path,
                        "detections_count": len(patch_detections)
                    }
                    patch_info.append(current_patch_info)
                    
                    # Vẽ và lưu patch nếu có phát hiện hoặc nếu cần lưu tất cả patch
                    if len(patch_detections) > 0 or not save_only_patches_with_detections:
                        # Vẽ các box lên patch
                        draw_boxes_on_patch(patch, patch_detections, x, y)
                        
                        # Lưu patch với kết quả
                        patch.save(patch_path)
                        print(f"Lưu patch {patch_id} với {len(patch_detections)} quảng cáo")
                except Exception as e:
                    print(f"Lỗi khi dự đoán với YOLOv8 cho patch {patch_id}: {str(e)}")
                
                # Cập nhật thanh tiến trình
                progress_bar.update(1)
        
        # Đóng thanh tiến trình
        progress_bar.close()
        
        # Xóa file tạm
        if os.path.exists(os.path.join(output_dir, "temp_patch.jpg")):
            os.remove(os.path.join(output_dir, "temp_patch.jpg"))
        
        # Áp dụng NMS cho toàn bộ kết quả
        filtered_results = non_max_suppression(all_results, nms_iou_threshold)
        print(f"Phát hiện {len(all_results)} đối tượng trước khi lọc, {len(filtered_results)} đối tượng sau khi lọc NMS")
        
        # Tạo thông tin tổng hợp cho ảnh này
        image_result = {
            "image_path": image_path,
            "image_size": {"width": width, "height": height},
            "total_patches": len(patch_info),
            "patches_with_detections": sum(1 for p in patch_info if p["detections_count"] > 0),
            "patch_info": patch_info,
            "detections": filtered_results
        }
        
        # Lưu thông tin tổng hợp
        summary_path = os.path.join(image_patches_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(image_result, f, indent=2, ensure_ascii=False)
        
        # Tạo ảnh tổng quan các patch (thumbnail map)
        create_patch_overview(image_path, patch_info, image_patches_dir)
        
        return filtered_results, patch_info
        
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []

# === Hàm tạo ảnh tổng quan các patch ===
def create_patch_overview(image_path, patch_info, output_folder):
    """Tạo ảnh tổng quan vị trí của các patch và phát hiện"""
    try:
        # Đọc kích thước ảnh gốc
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Tính kích thước của ảnh tổng quan (thu nhỏ)
        max_overview_size = 1000
        scale_factor = min(max_overview_size / img_width, max_overview_size / img_height)
        overview_width = int(img_width * scale_factor)
        overview_height = int(img_height * scale_factor)
        
        # Tạo ảnh tổng quan
        overview = Image.new('RGB', (overview_width, overview_height), (240, 240, 240))
        draw = ImageDraw.Draw(overview)
        
        # Cố gắng tải font, nếu không có thì dùng default
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except IOError:
            font = ImageFont.load_default()
        
        # Vẽ các patch lên ảnh tổng quan
        for patch in patch_info:
            # Tính tọa độ thu nhỏ của patch
            x = int(patch['x'] * scale_factor)
            y = int(patch['y'] * scale_factor)
            w = int(patch['width'] * scale_factor)
            h = int(patch['height'] * scale_factor)
            
            # Màu sắc dựa trên số lượng phát hiện
            if patch['detections_count'] > 0:
                # Màu càng đậm khi có nhiều phát hiện
                intensity = min(255, 100 + patch['detections_count'] * 50)
                color = (intensity, 255 - intensity, 0)  # Từ vàng đến đỏ
                outline_color = (255, 0, 0)  # Viền đỏ
            else:
                color = (200, 200, 200)  # Xám nhạt cho patch không có phát hiện
                outline_color = (150, 150, 150)  # Viền xám
            
            # Vẽ patch
            draw.rectangle([x, y, x+w, y+h], fill=color, outline=outline_color, width=1)
            
            # Vẽ ID của patch và số lượng phát hiện
            label = f"{patch['patch_id']}:{patch['detections_count']}"
            draw.text((x+2, y+2), label, fill=(0, 0, 0), font=font)
        
        # Lưu ảnh tổng quan
        overview_path = os.path.join(output_folder, "patch_overview.png")
        overview.save(overview_path)
        print(f"Đã tạo ảnh tổng quan tại: {overview_path}")
        
        return overview_path
    
    except Exception as e:
        print(f"Lỗi khi tạo ảnh tổng quan: {str(e)}")
        return None

# === Hàm lưu kết quả dạng JSON ===
def save_results_to_json(detections, output_path):
    """Lưu kết quả phát hiện thành file JSON"""
    try:
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"predictions": detections}, f, indent=2, ensure_ascii=False)
        print(f"Đã lưu kết quả JSON vào {output_path}")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu file JSON: {str(e)}")
        return False

# === Hàm xử lý một file ảnh ===
def process_image(image_path, model):
    """Xử lý một file ảnh đơn lẻ"""
    try:
        print(f"\n===== Đang xử lý ảnh: {image_path} =====")
        
        # Tạo đường dẫn đầu ra tương đối để giữ cấu trúc thư mục
        rel_path = os.path.relpath(image_path, input_folder)
        base_filename = os.path.splitext(rel_path)[0]
        
        # Tạo đường dẫn đầu ra đầy đủ cho JSON tổng hợp
        json_output_path = os.path.join(output_dir, f"{base_filename}_results.json")
        
        # Phát hiện quảng cáo và lưu các patch
        detections, patch_info = sliding_window_detect(image_path, model)
        
        if not detections:
            print(f"Không phát hiện quảng cáo nào trong ảnh {image_path}")
            return False
        
        # Lưu kết quả JSON tổng hợp
        save_results_to_json(detections, json_output_path)
        
        print(f"Đã xử lý xong ảnh {image_path}: {len(detections)} quảng cáo qua {len(patch_info)} patches")
        return True
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# === Hàm chính ===
def main():
    """Hàm chính thực thi toàn bộ quy trình"""
    # Kiểm tra thư mục đầu vào
    if not os.path.exists(input_folder):
        print(f"Không tìm thấy thư mục: {input_folder}")
        return
    
    # Tải mô hình YOLOv8
    model = load_model()
    
    # Tìm tất cả các file ảnh trong thư mục
    print(f"Đang tìm kiếm ảnh trong thư mục {input_folder}...")
    image_files = find_all_images(input_folder)
    
    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong thư mục {input_folder}")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh cần xử lý")
    
    # Tạo thư mục đầu ra
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục đầu ra: {output_dir}")
    
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)
        print(f"Đã tạo thư mục lưu patches: {patches_dir}")
    
    # Xử lý từng ảnh
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files):
        print(f"\nXử lý ảnh {i+1}/{len(image_files)}: {image_path}")
        
        try:
            result = process_image(image_path, model)
            
            if result:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Lỗi nghiêm trọng khi xử lý ảnh {image_path}: {str(e)}")
            failed += 1
            
            # Nếu gặp lỗi OutOfMemoryError, thử giải phóng bộ nhớ
            if "out of memory" in str(e).lower():
                print("Phát hiện lỗi hết bộ nhớ, đang cố gắng giải phóng...")
                import gc
                gc.collect()
    
    # Báo cáo kết quả
    print("\n===== KẾT QUẢ TỔNG HỢP =====")
    print(f"Tổng số ảnh đã xử lý: {len(image_files)}")
    print(f"- Thành công: {successful}")
    print(f"- Thất bại: {failed}")
    print(f"Kết quả được lưu trong thư mục: {output_dir}")
    print(f"Các patch được lưu trong thư mục: {patches_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình theo yêu cầu của người dùng.")
    except Exception as e:
        print(f"\nLỗi không mong muốn: {str(e)}")
        import traceback
        traceback.print_exc()