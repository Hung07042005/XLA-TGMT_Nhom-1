import cv2
import os

# ==================== THAY ĐƯỜNG DẪN ẢNH CỦA BẠN Ở ĐÂY ====================
img_path = 'photo.jpg'          # ←←←←← THAY BẰNG ẢNH MÀU CỦA BẠN
# ======================================================================

# Đọc ảnh (màu để dễ so sánh dung lượng file)
img = cv2.imread(img_path)

if img is None:
    print("❌ Không đọc được ảnh! Kiểm tra lại đường dẫn.")
else:
    print(f"✅ Shape ảnh gốc: {img.shape} (W x H x Channels)")

    # Giảm kích thước bằng slicing (theo đúng yêu cầu bài)
    img_half = img[::2, ::2]
    print(f"✅ Shape ảnh 1/2: {img_half.shape}")

    img_quarter = img[::4, ::4]
    print(f"✅ Shape ảnh 1/4: {img_quarter.shape}")

    # Lưu kết quả
    cv2.imwrite('half.jpg', img_half)
    cv2.imwrite('quarter.jpg', img_quarter)
    print("✅ Đã lưu: half.jpg và quarter.jpg")

    # So sánh dung lượng file (KB) - chỉ áp dụng với ảnh màu
    if len(img.shape) == 3:   # ảnh màu
        size_orig = os.path.getsize(img_path) / 1024
        size_half = os.path.getsize('half.jpg') / 1024
        size_quarter = os.path.getsize('quarter.jpg') / 1024

        print(f"\n📊 So sánh dung lượng file:")
        print(f"   Ảnh gốc     : {size_orig:.2f} KB")
        print(f"   Ảnh 1/2     : {size_half:.2f} KB")
        print(f"   Ảnh 1/4     : {size_quarter:.2f} KB")
        print(f"   → Ảnh 1/4 nhỏ hơn ảnh gốc khoảng {size_orig/size_quarter:.1f} lần")
    else:
        print("\nẢnh grayscale → không yêu cầu so sánh dung lượng.")
    
    cv2.imshow('Original', img)
    cv2.imshow('1/2', img_half)
    cv2.imshow('1/4', img_quarter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()