import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def adjust_brightness(image_path):
    # 1. Đọc ảnh dưới dạng grayscale [cite: 762]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Không tìm thấy ảnh!")
        return

    # 2. Tính toán thống kê histogram 
    mean_val = np.mean(img)
    std_val = np.std(img)
    
    corrected_img = img.copy()
    status = ""

    # 3. Logic Tự động phát hiện và sửa lỗi 
    # Trường hợp ảnh tối -> Gamma Correction (gamma < 1) 
    if mean_val < 80:
        gamma = 0.5
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_img = cv2.LUT(img, table)
        status = f"Dark detected (Mean={mean_val:.1f}). Applied Gamma {gamma}"

    # Trường hợp ảnh quá sáng -> Inverse Gamma (gamma > 1) 
    elif mean_val > 180:
        gamma = 2.2
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_img = cv2.LUT(img, table)
        status = f"Bright detected (Mean={mean_val:.1f}). Applied Inverse Gamma {gamma}"

    # Trường hợp tương phản thấp -> Histogram Equalization 
    elif std_val < 50:
        corrected_img = cv2.equalizeHist(img)
        status = f"Low Contrast detected (Std={std_val:.1f}). Applied Hist Equalization"
    
    else:
        status = "Image exposure looks normal."

    # 4. Hiển thị kết quả [cite: 769]
    plt.figure(figsize=(12, 8))

    # Hiển thị ảnh trước/sau
    plt.subplot(2, 2, 1); plt.imshow(img, cmap='gray'); plt.title("Original Image")
    plt.subplot(2, 2, 2); plt.imshow(corrected_img, cmap='gray'); plt.title("Corrected Image")

    # Hiển thị histogram trước/sau [cite: 769]
    plt.subplot(2, 2, 3); plt.hist(img.ravel(), 256, [0, 256]); plt.title("Original Histogram")
    plt.subplot(2, 2, 4); plt.hist(corrected_img.ravel(), 256, [0, 256]); plt.title("Corrected Histogram")

    plt.tight_layout()
    print(status)
    plt.show()

if __name__ == "__main__":
    # Tìm các file ảnh trong thư mục hiện tại
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    images = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]
    
    if not images:
        print("Không tìm thấy tệp ảnh nào trong thư mục hiện tại!")
    else:
        print("\n=== DANH SÁCH ẢNH DEMO ===")
        for idx, img_name in enumerate(images, 1):
            print(f"{idx}. {img_name}")
        
        try:
            choice = int(input(f"\nChọn số thứ tự ảnh (1-{len(images)}): "))
            if 1 <= choice <= len(images):
                selected_img = images[choice - 1]
                print(f"\nĐang xử lý ảnh: {selected_img}...")
                adjust_brightness(selected_img)
            else:
                print("Lựa chọn không hợp lệ!")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ!")