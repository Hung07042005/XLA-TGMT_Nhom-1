import cv2
import numpy as np
import argparse
from pathlib import Path


# ───────────────────────────────────────────────
# Gaussian noise
# ───────────────────────────────────────────────
def add_gaussian_noise(img, std=25):
    noise = np.random.normal(0, std, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────────
# Salt & Pepper noise (chuẩn cho detector)
# ───────────────────────────────────────────────
def add_salt_pepper_noise_strong(img, amount=0.05):
    noisy = img.copy()
    h, w = img.shape
    num_pixels = int(amount * h * w)

    coords_added = 0
    while coords_added < num_pixels:
        i = np.random.randint(1, h - 1)
        j = np.random.randint(1, w - 1)

        patch = img[i-1:i+2, j-1:j+2]
        median_val = np.median(patch)

        if abs(median_val - 0) > 50:
            noisy[i, j] = 0
            coords_added += 1
        elif abs(median_val - 255) > 50:
            noisy[i, j] = 255
            coords_added += 1

    return noisy


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate noisy images")
    parser.add_argument("input", help="Clean image path")
    parser.add_argument("--type", choices=["gaussian", "sp"], help="Noise type")
    parser.add_argument("--all", action="store_true", help="Generate all noise types")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot load image")

    input_path = Path(args.input)

    # ───────────────
    # Generate ALL
    # ───────────────
    if args.all:
        noisy_g = add_gaussian_noise(img)
        out_g = f"{input_path.stem}_gaussian.png"
        cv2.imwrite(out_g, noisy_g)
        print(f"Saved: {out_g}")

        noisy_sp = add_salt_pepper_noise_strong(img)
        out_sp = f"{input_path.stem}_sp.png"
        cv2.imwrite(out_sp, noisy_sp)
        print(f"Saved: {out_sp}")

        return

    # ───────────────
    # Single type
    # ───────────────
    if args.type == "gaussian":
        noisy = add_gaussian_noise(img)
        out_name = args.output or f"{input_path.stem}_gaussian.png"

    elif args.type == "sp":
        noisy = add_salt_pepper_noise_strong(img)
        out_name = args.output or f"{input_path.stem}_sp.png"

    else:
        raise ValueError("Please use --type or --all")

    cv2.imwrite(out_name, noisy)
    print(f"Saved: {out_name}")


if __name__ == "__main__":
    main()