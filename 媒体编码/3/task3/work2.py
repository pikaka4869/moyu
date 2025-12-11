import numpy as np
from scipy.fftpack import dct, idct
from scipy.fft import fft2, ifft2
from PIL import Image

# -------------------------------
# 1. 读取灰度图
# -------------------------------
img = Image.open("lena.png").convert("L")
img_gray = np.array(img, dtype=np.float32)

# -------------------------------
# 2. DCT / 整数 DCT / DFT / KLT
# -------------------------------
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def int_dct2(block, scale=16384):
    coeff = dct2(block)
    return np.round(coeff*scale).astype(int), scale

def int_idct2(coeff, scale):
    return idct2(coeff.astype(float)/scale)

def dft2(block):
    return fft2(block)

def idft2(coeff):
    return np.real(ifft2(coeff))

def klt(blocks):
    data = blocks.reshape(len(blocks), -1)
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx], mean

def apply_klt(block, transform_matrix, mean):
    vec = block.flatten() - mean
    return transform_matrix.T @ vec

def inverse_klt(coeff, transform_matrix, mean):
    return (transform_matrix @ coeff + mean).reshape(8,8)

# -------------------------------
# 3. JPEG标准亮度量化表
# -------------------------------
std_qtable = np.array([
 [16,11,10,16,24,40,51,61],
 [12,12,14,19,26,58,60,55],
 [14,13,16,24,40,57,69,56],
 [14,17,22,29,51,87,80,62],
 [18,22,37,56,68,109,103,77],
 [24,35,55,64,81,104,113,92],
 [49,64,78,87,103,121,120,101],
 [72,92,95,98,112,100,103,99]
], dtype=np.float32)

qtable_half = std_qtable*0.5
qtable_double = std_qtable*2.0

# -------------------------------
# 4. PSNR计算
# -------------------------------
def psnr(orig, recon):
    mse = np.mean((orig - recon)**2)
    return 10*np.log10(255**2 / mse)

# -------------------------------
# 5. 块处理 + 详细日志
# -------------------------------
log_file = open("full_process_log.txt", "w")

h, w = img_gray.shape
recon_img = np.zeros_like(img_gray)

for i in range(0,h,8):
    for j in range(0,w,8):
        block = img_gray[i:i+8, j:j+8]
        if block.shape != (8,8):
            continue
        
        log_file.write(f"\n--- 块位置 ({i},{j}) ---\n")
        log_file.write(f"原块:\n{block}\n")

        # --- DCT ---
        dct_coeff = dct2(block-128)
        log_file.write(f"DCT系数:\n{dct_coeff}\n")

        # --- 整数 DCT ---
        int_coeff, scale = int_dct2(block-128)
        log_file.write(f"整数DCT系数 (scale={scale}):\n{int_coeff}\n")

        # --- 量化 (标准表) ---
        q_block = np.round(dct_coeff / std_qtable)
        log_file.write(f"量化后系数:\n{q_block}\n")

        # --- 反量化 + 反DCT ---
        recon_block = idct2(q_block*std_qtable) + 128
        log_file.write(f"反变换重建块:\n{recon_block}\n")

        # --- 块误差 ---
        err = np.abs(block - recon_block)
        log_file.write(f"块最大误差: {np.max(err)}\n")
        log_file.write(f"块平均误差: {np.mean(err)}\n")

        # 写入重建图像
        recon_img[i:i+8,j:j+8] = recon_block

# 保存重建图像
recon_img = np.clip(recon_img,0,255).astype(np.uint8)
Image.fromarray(recon_img).save("recon_full_log.png")

# 全图 PSNR
full_psnr = psnr(img_gray, recon_img)
log_file.write(f"\n全图 PSNR: {full_psnr:.2f} dB\n")
log_file.close()
print(f"全流程仿真完成, 全图 PSNR={full_psnr:.2f} dB, 日志保存为 full_process_log.txt")
