import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


# ==================================================
# 1. Gaussian Kernel
# ==================================================
def gaussian_kernel(ksize, sigma=None):
    if sigma is None:
        sigma = ksize / 6.0

    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


# ==================================================
# 2. Spatial Blur
# ==================================================
def spatial_blur(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# ==================================================
# 3. FFT Blur (FIXED - correct version)
# ==================================================
def fft_blur(img, ksize):
    if img.ndim == 3:
        channels = cv2.split(img)
        return cv2.merge([fft_blur(c, ksize) for c in channels])

    h, w = img.shape

    kernel = gaussian_kernel(ksize)

    # Pad kernel to image size
    kernel_padded = np.zeros((h, w))
    kernel_padded[:ksize, :ksize] = kernel

    # 🔥 QUAN TRỌNG: shift kernel về (0,0)
    kernel_padded = np.fft.ifftshift(kernel_padded)

    # FFT
    img_fft = np.fft.fft2(img.astype(np.float32))
    kernel_fft = np.fft.fft2(kernel_padded)

    result = np.fft.ifft2(img_fft * kernel_fft).real

    return np.clip(result, 0, 255).astype(np.uint8)


# ==================================================
# 4. Benchmark
# ==================================================
def benchmark(img, ksize, crop_size=100, runs=10):
    h, w = img.shape[:2]

    cy, cx = h // 2, w // 2
    crop = img[cy - crop_size//2: cy + crop_size//2,
               cx - crop_size//2: cx + crop_size//2]

    # Spatial
    t0 = time.perf_counter()
    for _ in range(runs):
        spatial_blur(crop, ksize)
    t_spatial = (time.perf_counter() - t0) / runs * 1000

    # FFT
    t0 = time.perf_counter()
    for _ in range(runs):
        fft_blur(crop, ksize)
    t_fft = (time.perf_counter() - t0) / runs * 1000

    return t_spatial, t_fft


# ==================================================
# 5. Smart Selector
# ==================================================
def smart_blur(img, ksize):
    if ksize % 2 == 0:
        ksize += 1

    t_spatial, t_fft = benchmark(img, ksize)

    # Decision
    if t_spatial < t_fft:
        method = "Spatial"
        func = spatial_blur
        reason = f"Spatial faster ({t_spatial:.3f} ms < {t_fft:.3f} ms)"
    else:
        method = "FFT"
        func = fft_blur
        reason = f"FFT faster ({t_fft:.3f} ms < {t_spatial:.3f} ms)"

    # Apply full image
    t0 = time.perf_counter()
    result = func(img, ksize)
    full_time = (time.perf_counter() - t0) * 1000

    report = {
        "method": method,
        "ksize": ksize,
        "t_spatial": t_spatial,
        "t_fft": t_fft,
        "t_full": full_time,
        "reason": reason
    }

    return result, report


# ==================================================
# 6. Visualization
# ==================================================
def show_results(original, blurred, report):
    plt.figure(figsize=(12, 6))

    # Original
    plt.subplot(1, 2, 1)
    if original.ndim == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Blurred
    plt.subplot(1, 2, 2)
    if blurred.ndim == 3:
        plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(blurred, cmap='gray')
    plt.title(f"{report['method']} (k={report['ksize']})\n{report['t_full']:.2f} ms")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ==================================================
# 7. Main
# ==================================================
if __name__ == "__main__":
    img = cv2.imread("input.jpg")  # đổi tên ảnh của bạn
    if img is None:
        raise FileNotFoundError("Không tìm thấy ảnh")

    ksize = 51  # thử 5, 15, 31, 51, 101

    blurred, report = smart_blur(img, ksize)

    # ===== REPORT =====
    print("\n" + "="*60)
    print("SMART BLUR SELECTOR REPORT")
    print("="*60)
    print(f"Kernel size     : {report['ksize']}")
    print(f"Method chosen   : {report['method']}")
    print(f"Spatial time    : {report['t_spatial']:.3f} ms")
    print(f"FFT time        : {report['t_fft']:.3f} ms")
    print(f"Full image time : {report['t_full']:.3f} ms")
    print(f"Reason          : {report['reason']}")
    print("="*60)

    # Save
    cv2.imwrite("blurred_output.png", blurred)
    print("✅ Saved: blurred_output.png")

    # Show
    show_results(img, blurred, report)