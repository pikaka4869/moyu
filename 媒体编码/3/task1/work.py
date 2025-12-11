import numpy as np
from scipy.fftpack import dct, idct
from scipy.fft import fft2, ifft2
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# 1. 读取图像并切分 8x8 块
# -------------------------------
img = Image.open("lena.png").convert("L")  # 灰度图
img_gray = np.array(img) / 255.0

def img_to_blocks(img, block_size=8):
    h, w = img.shape
    blocks = []
    for i in range(0, h - h % block_size, block_size):
        for j in range(0, w - w % block_size, block_size):
            blocks.append(img[i:i+block_size, j:j+block_size])
    return np.array(blocks)  # (N_blocks, 8, 8)

blocks = img_to_blocks(img_gray)
block = blocks[0]

# -------------------------------
# 2. DCT 与整数 DCT
# -------------------------------
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(coeff):
    return idct(idct(coeff.T, norm='ortho').T, norm='ortho')

def int_dct2(block, scale=16384):
    coeff = dct2(block)
    return np.round(coeff * scale).astype(int), scale

def int_idct2(coeff, scale):
    return idct2(coeff.astype(float)/scale)

# -------------------------------
# 3. DFT
# -------------------------------
def dft2(block):
    return fft2(block)

def idft2(coeff):
    return np.real(ifft2(coeff))

# -------------------------------
# 4. KLT
# -------------------------------
def klt(blocks):
    data = blocks.reshape(len(blocks), -1)
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    cov = np.cov(data_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx], mean

def apply_klt(block, transform_matrix, mean):
    vec = block.flatten() - mean
    return transform_matrix.T @ vec

def inverse_klt(coeff, transform_matrix, mean):
    return (transform_matrix @ coeff + mean).reshape(8,8)

# -------------------------------
# 5. 高频系数占比
# -------------------------------
def high_freq_ratio(coeff, threshold=0.1):
    return np.sum(np.abs(coeff) > threshold) / coeff.size

# -------------------------------
# 6. PSNR
# -------------------------------
def psnr(orig, recon):
    mse = np.mean((orig - recon)**2)
    return 10 * np.log10(1 / mse)

# -------------------------------
# 7. 变换 - 反变换
# -------------------------------
dct_coeff = dct2(block)
dct_recon = idct2(dct_coeff)

int_dct_coeff, scale = int_dct2(block)
int_dct_recon = int_idct2(int_dct_coeff, scale)

dft_coeff = dft2(block)
dft_recon = idft2(dft_coeff)

klt_matrix, klt_mean = klt(blocks)
klt_coeff = apply_klt(block, klt_matrix, klt_mean)
klt_recon = inverse_klt(klt_coeff, klt_matrix, klt_mean)

# -------------------------------
# 8. 输出指标
# -------------------------------
print("高频系数占比:")
print(f"DCT: {high_freq_ratio(dct_coeff):.3f}")
print(f"Int DCT: {high_freq_ratio(int_dct_coeff/scale):.3f}")
print(f"DFT: {high_freq_ratio(dft_coeff):.3f}")
print(f"KLT: {high_freq_ratio(klt_coeff):.3f}")

print("\n反变换 PSNR:")
print(f"DCT: {psnr(block, dct_recon):.2f} dB")
print(f"Int DCT: {psnr(block, int_dct_recon):.2f} dB")
print(f"DFT: {psnr(block, dft_recon):.2f} dB")
print(f"KLT: {psnr(block, klt_recon):.2f} dB")

# -------------------------------
# 9. 热力图可视化
# -------------------------------
plt.figure(figsize=(10,2))
for i, (name, coeff) in enumerate([
    ('DCT', dct_coeff),
    ('Int DCT', int_dct_coeff/scale),
    ('DFT', np.abs(dft_coeff)),
    ('KLT', np.abs(klt_coeff.reshape(8,8)))
]):
    plt.subplot(1,4,i+1)
    plt.imshow(coeff, cmap='hot', interpolation='nearest')
    plt.title(name)
    plt.colorbar()
plt.savefig("coeff_heatmap.png")
