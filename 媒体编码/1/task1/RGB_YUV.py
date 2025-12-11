import cv2
import numpy as np
import os
import csv
from math import log10
import matplotlib.pyplot as plt

INPUT_IMAGE = "test.png"   
OUT_DIR = "experiment_results"
Q_STEPS = [1, 2, 4, 8, 16, 32]
SAMPLING_MODES = ["444", "422", "420"]
SAVE_RECON_PER_COMBO = True 
USE_BILINEAR_UPSAMPLE = False  
# -------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "recon"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)


# -----------------------------
# 基础工具
# -----------------------------
def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 99.0
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / (mse ** 0.5))


def crop_to_even_wh(img):
    """裁剪 H 和 W 到偶数（向下裁剪）。返回裁剪后图像和新 H,W。"""
    h, w = img.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    return img[:h2, :w2], h2, w2


def ensure_width_even(img):
    """裁剪宽度到偶数（保持高度）。返回裁剪后图像和新 H,W。"""
    h, w = img.shape[:2]
    w2 = w - (w % 2)
    return img[:, :w2], h, w2


# -----------------------------
# RGB <-> YUV (BT.601) （浮点）
# -----------------------------
def rgb_to_yuv444(rgb):
    R = rgb[:, :, 2].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 0].astype(np.float32)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.169 * R - 0.331 * G + 0.5 * B + 128.0
    V = 0.5 * R - 0.419 * G - 0.081 * B + 128.0
    return Y, U, V


def yuv444_to_rgb(Y, U, V):
    Uc = U - 128.0
    Vc = V - 128.0
    R = Y + 1.402 * Vc
    G = Y - 0.34414 * Uc - 0.71414 * Vc
    B = Y + 1.772 * Uc
    rgb = np.stack([B, G, R], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


# -----------------------------
# 采样 / 下采样
# -----------------------------
def downsample_422(U):
    """
    4:2:2 — 水平方向减半（对每行按相邻两列平均）。
    注意：要求宽度为偶数（函数内部不裁剪）。
    """
    # shape H x W -> H x (W/2)
    return 0.5 * (U[:, 0::2] + U[:, 1::2])


def upsample_422_nearest(U422, H, W):
    """
    最近邻恢复 4:2:2 -> 4:4:4（水平复制）
    U422: H x (W/2)
    """
    W_even = W - (W % 2)
    U = np.zeros((H, W_even), dtype=np.float32)
    U[:, 0::2] = U422
    U[:, 1::2] = U422
    if W != W_even:
        U = np.hstack([U, U[:, -1:]])
    return U


def downsample_420(U):
    """
    4:2:0 下采样（2x2 块平均）
    要求 H 和 W 为偶数（函数内部会裁剪到偶数）
    """
    U = U[:(U.shape[0] - U.shape[0] % 2), :(U.shape[1] - U.shape[1] % 2)]
    return 0.25 * (U[0::2, 0::2] + U[1::2, 0::2] + U[0::2, 1::2] + U[1::2, 1::2])


def upsample_420_nearest(U420, H, W):
    """
    重建 4:2:0 -> 4:4:4（最近邻复制）
    U420: (H/2) x (W/2)
    """
    H_even = H - (H % 2)
    W_even = W - (W % 2)
    U = np.zeros((H_even, W_even), dtype=np.float32)
    U[0::2, 0::2] = U420
    U[1::2, 0::2] = U420
    U[0::2, 1::2] = U420
    U[1::2, 1::2] = U420
    if H != H_even:
        U = np.vstack([U, U[-1:]])
    if W != W_even:
        U = np.hstack([U, U[:, -1:]])
    return U


def upsample_via_interpolation_lowres_to_full(C_lowres, H, W):
    """
    如果选择双线性插值：把低分辨率 chroma 用 cv2.resize 放大回 HxW
    cv2.resize 的 size 参数为 (W, H).
    """
    return cv2.resize(C_lowres.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


# -----------------------------
# 量化（均匀标量量化）
# -----------------------------
def quantize_uniform(C, qstep):
    # qstep=1 表示不明显量化（只是 round）
    return (np.round(C / qstep) * qstep)


# -----------------------------
# 实验主循环
# -----------------------------
def run_experiment(img_path, q_steps, sampling_modes):
    orig = cv2.imread(img_path)
    if orig is None:
        raise FileNotFoundError(f"找不到文件: {img_path}")

    # 保留原始用于 PSNR 比较（uint8）
    orig_uint8 = orig.copy()
    # 裁剪为偶数 H/W（便于 420）
    orig_cropped, H, W = crop_to_even_wh(orig_uint8)
    print(f"原图尺寸: {orig_uint8.shape[0]}x{orig_uint8.shape[1]} -> 裁剪后 {H}x{W}")

    Y_full, U_full, V_full = rgb_to_yuv444(orig_cropped)

    results = []  

    for mode in sampling_modes:
        for q in q_steps:
            if mode == "444":
                Uq = quantize_uniform(U_full, q)
                Vq = quantize_uniform(V_full, q)
                U_rec = Uq
                V_rec = Vq

            elif mode == "422":
                # 需要保证宽度偶数
                U422 = downsample_422(U_full)
                V422 = downsample_422(V_full)
                U422q = quantize_uniform(U422, q)
                V422q = quantize_uniform(V422, q)
                if USE_BILINEAR_UPSAMPLE:
                    U_rec = upsample_via_interpolation_lowres_to_full(U422q, H, W)
                    V_rec = upsample_via_interpolation_lowres_to_full(V422q, H, W)
                else:
                    U_rec = upsample_422_nearest(U422q, H, W)
                    V_rec = upsample_422_nearest(V422q, H, W)

            elif mode == "420":
                U420 = downsample_420(U_full)
                V420 = downsample_420(V_full)
                U420q = quantize_uniform(U420, q)
                V420q = quantize_uniform(V420, q)
                if USE_BILINEAR_UPSAMPLE:
                    U_rec = upsample_via_interpolation_lowres_to_full(U420q, H, W)
                    V_rec = upsample_via_interpolation_lowres_to_full(V420q, H, W)
                else:
                    U_rec = upsample_420_nearest(U420q, H, W)
                    V_rec = upsample_420_nearest(V420q, H, W)
            else:
                raise ValueError("未知采样模式: " + mode)

            rgb_rec = yuv444_to_rgb(Y_full, U_rec, V_rec)

            ps = psnr(orig_cropped, rgb_rec)

            if SAVE_RECON_PER_COMBO:
                out_name = f"recon_{mode}_Q{q}.png"
                cv2.imwrite(os.path.join(OUT_DIR, "recon", out_name), rgb_rec)

            print(f"mode={mode}, Q={q} -> PSNR={ps:.3f} dB")
            results.append((mode, q, float(ps)))

    csv_path = os.path.join(OUT_DIR, "psnr_results.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "Q", "PSNR_dB"])
        for r in results:
            writer.writerow(r)
    print("结果已保存到:", csv_path)

    plt.figure(figsize=(8, 5))
    for mode in sampling_modes:
        mode_rows = [(q, ps) for (m, q, ps) in results if m == mode]
        mode_rows = sorted(mode_rows, key=lambda x: x[0])
        qs = [q for q, _ in mode_rows]
        pss = [ps for _, ps in mode_rows]
        plt.plot(qs, pss, marker='o', label=mode)
    plt.xlabel("Quantization step (Q)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR vs Q - {os.path.basename(img_path)}")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(OUT_DIR, "plots", "psnr_vs_q.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("曲线已保存到:", plot_path)

    return results


if __name__ == "__main__":
    results = run_experiment(INPUT_IMAGE, Q_STEPS, SAMPLING_MODES)
