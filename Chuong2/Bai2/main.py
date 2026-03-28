import cv2
import numpy as np
import time
from scipy.signal import fftconvolve


# TẠO KERNEL GAUSSIAN

def tao_kernel_gaussian(kich_thuoc, sigma=1):
    truc = np.linspace(-(kich_thuoc // 2), kich_thuoc // 2, kich_thuoc)
    xx, yy = np.meshgrid(truc, truc)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)



# PHƯƠNG PHÁP 1: Spatial

def lam_mo_spatial(anh, kernel):
    return cv2.filter2D(anh, -1, kernel)



# PHƯƠNG PHÁP 2: FFT

def lam_mo_fft(anh, kernel):
    ket_qua = np.zeros_like(anh)

    # xử lý từng kênh màu (R, G, B)
    for kenh in range(anh.shape[2]):
        ket_qua[:, :, kenh] = fftconvolve(anh[:, :, kenh], kernel, mode='same')

    return ket_qua.astype(np.uint8)



# HÀM ĐO THỜI GIAN

def do_thoi_gian(ham, anh, kernel):
    bat_dau = time.time()
    ham(anh, kernel)
    ket_thuc = time.time()
    return ket_thuc - bat_dau

def lam_mo_thong_minh(duong_dan_anh, kich_thuoc_kernel):
    # đọc ảnh
    anh = cv2.imread(duong_dan_anh)
    anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)

    # tạo kernel
    kernel = tao_kernel_gaussian(kich_thuoc_kernel, sigma=kich_thuoc_kernel / 6)

    # cắt vùng nhỏ 100x100 để test
    anh_nho = anh[0:100, 0:100]

    # đo thời gian 2 phương pháp
    time_spatial = do_thoi_gian(lam_mo_spatial, anh_nho, kernel)
    time_fft = do_thoi_gian(lam_mo_fft, anh_nho, kernel)

    # chọn phương pháp nhanh hơn
    if time_spatial < time_fft:
        phuong_phap = "Spatial"
        ket_qua = lam_mo_spatial(anh, kernel)
        thoi_gian = time_spatial
    else:
        phuong_phap = "FFT (biến đổi Fourier)"
        ket_qua = lam_mo_fft(anh, kernel)
        thoi_gian = time_fft


    print("===== BÁO CÁO =====")
    print(f"Kích thước kernel: {kich_thuoc_kernel}")
    print(f"Thời gian Spatial: {time_spatial:.6f} giây")
    print(f"Thời gian FFT: {time_fft:.6f} giây")
    print(f"Phương pháp được chọn: {phuong_phap}")
    print(f"Thời gian thực thi: {thoi_gian:.6f} giây")

    if "FFT" in phuong_phap:
        print("Giải thích: Kernel lớn nên FFT nhanh hơn")
    else:
        print("Giải thích: Kernel nhỏ nên Spatial nhanh hơn")

    return ket_qua


# CHẠY CHƯƠNG TRÌNH

ket_qua = lam_mo_thong_minh("input.jpg", 31)

# lưu ảnh kết quả
cv2.imwrite("output.jpg", cv2.cvtColor(ket_qua, cv2.COLOR_RGB2BGR))