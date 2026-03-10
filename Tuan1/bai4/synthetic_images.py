import numpy as np
import cv2

# Kích thước ảnh
height = 256
width  = 256

# =================================================================
# Ảnh 1: Gradient ngang từ đen (0) sang trắng (255)
# =================================================================
gradient = np.linspace(0, 255, width, dtype=np.uint8)   # mảng 1D: [0,1,2,...,255]
gradient_horizontal = np.tile(gradient, (height, 1))     # lặp lại thành ma trận 256×256

# =================================================================
# Ảnh 2: Checkerboard (ô vuông trắng-đen xen kẽ)
# =================================================================
# Cách 1: đơn giản, ô 32×32 (chia hết cho 256)
checker_size = 32
checker = np.zeros((height, width), dtype=np.uint8)

# Tạo pattern lặp lại
for i in range(height):
    for j in range(width):
        # Nếu (i//checker_size + j//checker_size) chẵn → trắng (255), lẻ → đen (0)
        if (i // checker_size + j // checker_size) % 2 == 0:
            checker[i, j] = 255
        else:
            checker[i, j] = 0

# Cách ngắn hơn (vectorized - nhanh & đẹp):
# checker = ((np.arange(height)[:,None] // 32 + np.arange(width) // 32) % 2) * 255
# checker = checker.astype(np.uint8)

# =================================================================
# Ảnh 3: Vòng tròn trắng trên nền đen
# =================================================================
circle = np.zeros((height, width), dtype=np.uint8)

# Tâm vòng tròn ở giữa ảnh
center_x = width // 2
center_y = height // 2
radius = 80               # bán kính vòng tròn (thay đổi tùy ý)

# Tạo lưới tọa độ
y, x = np.ogrid[0:height, 0:width]
distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

# Pixel nào nằm trong vòng tròn thì = 255
circle[distance_from_center <= radius] = 255

# =================================================================
# Lưu file và in thông tin kiểm tra
# =================================================================
images = {
    'gradient_horizontal.jpg': gradient_horizontal,
    'checkerboard.jpg'       : checker,
    'circle.jpg'             : circle
}

for filename, img in images.items():
    cv2.imwrite(filename, img)
    print(f"Đã lưu: {filename}")
    print(f"  • shape   : {img.shape}")
    print(f"  • dtype   : {img.dtype}")
    print(f"  • min/max : {img.min()} – {img.max()}\n")

# ----------------- RGB version -----------------
# Cách 1: 3 kênh giống nhau (ảnh xám thành màu RGB)
gradient_rgb = np.stack([gradient_horizontal]*3, axis=-1)   # shape: (256,256,3)

# Cách 2: mỗi kênh một pattern khác nhau (đẹp, thú vị)
rgb_multi = np.zeros((height, width, 3), dtype=np.uint8)
rgb_multi[:,:,0] = gradient_horizontal          # Red   ← gradient ngang
rgb_multi[:,:,1] = checker                      # Green ← checkerboard
rgb_multi[:,:,2] = circle                       # Blue  ← vòng tròn

cv2.imwrite('gradient_rgb.jpg', gradient_rgb)
cv2.imwrite('multi_pattern_rgb.jpg', rgb_multi)

print("Bonus RGB:")
print("  gradient_rgb.jpg        → gradient xám trên 3 kênh")
print("  multi_pattern_rgb.jpg   → mỗi kênh một pattern khác nhau")
print("Hoàn tất! Mở 3 file ảnh để kiểm tra kết quả.")