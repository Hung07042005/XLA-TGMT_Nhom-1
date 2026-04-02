import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. Analyze Exposure
# ==================================================
def analyze_exposure(img):
    mean_val = np.mean(img)
    std_val = np.std(img)

    if mean_val < 80:
        return "dark", mean_val, std_val
    elif mean_val > 180:
        return "bright", mean_val, std_val
    elif std_val < 50:
        return "low_contrast", mean_val, std_val
    else:
        return "normal", mean_val, std_val


# ==================================================
# 2. Gamma Correction
# ==================================================
def gamma_correction(img, gamma):
    table = np.array([
        ((i / 255.0) ** gamma) * 255 for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


# ==================================================
# 3. Contrast Enhancement (CLAHE)
# ==================================================
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# ==================================================
# 4. Main Function
# ==================================================
def adaptive_brightness_adjuster(image_path):

    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Không tìm thấy ảnh!")

    # Analyze
    status, mean_val, std_val = analyze_exposure(img)

    # Apply correction
    if status == "dark":
        gamma = 0.5
        corrected = gamma_correction(img, gamma)
        method = f"Gamma Correction (γ={gamma})"

    elif status == "bright":
        gamma = 2.0
        corrected = gamma_correction(img, gamma)
        method = f"Inverse Gamma (γ={gamma})"

    elif status == "low_contrast":
        corrected = apply_clahe(img)
        method = "CLAHE (Histogram Equalization)"

    else:
        corrected = img.copy()
        method = "No correction needed"

    # ==================================================
    # REPORT
    # ==================================================
    print("\n" + "="*60)
    print("ADAPTIVE BRIGHTNESS ADJUSTER - REPORT")
    print("="*60)
    print(f"Detected condition : {status}")
    print(f"Mean intensity     : {mean_val:.2f}")
    print(f"Std deviation      : {std_val:.2f}")
    print(f"Method applied     : {method}")
    print("="*60)

    # ==================================================
    # Visualization
    # ==================================================
    plt.figure(figsize=(12, 8))

    # Images
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(corrected, cmap='gray')
    plt.title("Corrected")
    plt.axis('off')

    # Histograms
    plt.subplot(2, 2, 3)
    plt.hist(img.ravel(), bins=256, range=(0, 256))
    plt.title("Original Histogram")

    plt.subplot(2, 2, 4)
    plt.hist(corrected.ravel(), bins=256, range=(0, 256))
    plt.title("Corrected Histogram")

    plt.tight_layout()
    plt.show()

    # Save result
    cv2.imwrite("corrected_image.png", corrected)
    print("✅ Saved: corrected_image.png")

    return corrected


# ==================================================
# 5. MAIN
# ==================================================
if __name__ == "__main__":
    image_path = "image_dark.png"   # đổi ảnh tại đây
    adaptive_brightness_adjuster(image_path)