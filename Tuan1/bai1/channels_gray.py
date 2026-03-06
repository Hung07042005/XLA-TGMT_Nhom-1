import cv2
import numpy as np

# 1. Đọc ảnh màu
img = cv2.imread('photo.jpg') # Hãy đảm bảo bạn có file này trong thư mục

if img is None:
    print("Không tìm thấy file photo.jpg")
else:
    # 2. Tách 3 kênh B, G, R bằng NumPy indexing (OpenCV đọc ảnh theo thứ tự BGR)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    # Lưu từng kênh
    cv2.imwrite('blue.png', B)
    cv2.imwrite('green.png', G)
    cv2.imwrite('red.png', R)

    # 3. Tự cài đặt chuyển RGB -> grayscale (Chọn Option 2)
    # Lưu ý: Phải ép kiểu sang float trước khi tính toán để tránh tràn bộ nhớ (overflow) của uint8
    gray_float = 0.299 * R.astype(np.float32) + 0.587 * G.astype(np.float32) + 0.114 * B.astype(np.float32)
    
    # Ép kiểu ngược lại về uint8 để làm ảnh
    gray_manual = gray_float.astype(np.uint8)
    
    # Lưu ảnh
    cv2.imwrite('gray_manual.png', gray_manual)

    # 4. So sánh Shape
    print("--- So sánh Shape ---")
    print(f"Shape ảnh màu gốc: {img.shape}")
    print(f"Shape kênh Blue: {B.shape}")
    print(f"Shape kênh Green: {G.shape}")
    print(f"Shape kênh Red: {R.shape}")
    print(f"Shape ảnh grayscale: {gray_manual.shape}")