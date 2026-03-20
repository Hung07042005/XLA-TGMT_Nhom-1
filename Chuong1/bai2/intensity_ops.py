import os
import cv2
import numpy as np

if not os.path.exists('gray_manual.png'):
    # tạo ảnh 256×256 xám
    cv2.imwrite('gray_manual.png', np.full((256,256), 128, dtype=np.uint8))
    print('Tạo ảnh thử nghiệm gray_manual.png')

# Tự viết hàm clip_uint8 để đảm bảo giá trị trong [0, 255]
def clip_uint8(arr):
    # np.clip giới hạn các giá trị < 0 thành 0 và > 255 thành 255
    return np.clip(arr, 0, 255).astype(np.uint8)

# 1. Đọc ảnh grayscale (có thể dùng ảnh gray_manual.png từ Bài 1)
gray = cv2.imread('gray_manual.png', cv2.IMREAD_GRAYSCALE)

if gray is not None:
    # Ép kiểu sang int16 hoặc float để tính toán không bị lỗi khi vượt quá 255 hoặc nhỏ hơn 0
    gray_calc = gray.astype(np.int16)

    # 2. Tạo ảnh tối hơn, sáng hơn và tăng tương phản
    gray_dark = clip_uint8(gray_calc - 50)
    gray_bright = clip_uint8(gray_calc + 50)
    
    alpha = 1.5 # Hệ số tăng tương phản > 1
    gray_contrast = clip_uint8(gray_calc * alpha)

    # 3. Tự cài đặt threshold nhị phân (ngưỡng T = 128)
    T = 128
    # Dùng np.where: nếu pixel >= T thì gán 255, ngược lại gán 0
    gray_thresh = np.where(gray >= T, 255, 0).astype(np.uint8)

    # Lưu tất cả các ảnh
    cv2.imwrite('gray_dark.png', gray_dark)
    cv2.imwrite('gray_bright.png', gray_bright)
    cv2.imwrite('gray_contrast.png', gray_contrast)
    cv2.imwrite('gray_thresh.png', gray_thresh)

    # 4. In giá trị pixel trước và sau tại vị trí [100, 100]
    y, x = 100, 100
    print(f"--- Kiểm tra pixel tại vị trí [{y}, {x}] ---")
    print(f"Ảnh gốc: {gray[y, x]}")
    print(f"Ảnh tối hơn (-50): {gray_dark[y, x]}")
    print(f"Ảnh sáng hơn (+50): {gray_bright[y, x]}")
    print(f"Ảnh tăng tương phản (x1.5): {gray_contrast[y, x]}")
    print(f"Ảnh nhị phân (Threshold={T}): {gray_thresh[y, x]}")