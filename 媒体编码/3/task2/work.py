import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# -------------------------------
# 1. 读取灰度图
# -------------------------------
img = Image.open("lena.png").convert("L")
img_gray = np.array(img, dtype=np.float32)

# -------------------------------
# 2. JPEG 标准亮度量化表
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

# 生成步长缩小 / 放大量化表
qtable_half = std_qtable * 0.5
qtable_double = std_qtable * 2.0

# -------------------------------
# 3. DCT / IDCT 函数
# -------------------------------
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# -------------------------------
# 4. 块处理函数
# -------------------------------
def process_blocks(img, qtable):
    h, w = img.shape
    recon = np.zeros_like(img)
    total_coeff = 0
    nonzero_coeff = 0
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8]
            if block.shape != (8,8):
                continue
            dct_block = dct2(block - 128)  # DC shift
            q_block = np.round(dct_block / qtable)
            total_coeff += 64
            nonzero_coeff += np.sum(q_block != 0)
            recon[i:i+8, j:j+8] = idct2(q_block * qtable) + 128
    recon = np.clip(recon, 0, 255)
    bitrate = nonzero_coeff / total_coeff  # 非零系数比例近似码率
    return recon.astype(np.uint8), bitrate

# -------------------------------
# 5. 压缩 - 解压缩实验
# -------------------------------
recon_std, rate_std = process_blocks(img_gray, std_qtable)
recon_half, rate_half = process_blocks(img_gray, qtable_half)
recon_double, rate_double = process_blocks(img_gray, qtable_double)

# -------------------------------
# 6. PSNR
# -------------------------------
def psnr(orig, recon):
    mse = np.mean((orig - recon)**2)
    return 10 * np.log10(255**2 / mse)

print("PSNR / 码率:")
print(f"标准表: PSNR={psnr(img_gray,recon_std):.2f} dB, 码率={rate_std:.3f}")
print(f"步长缩小: PSNR={psnr(img_gray,recon_half):.2f} dB, 码率={rate_half:.3f}")
print(f"步长放大: PSNR={psnr(img_gray,recon_double):.2f} dB, 码率={rate_double:.3f}")

# -------------------------------
# 7. 保存重建图像
# -------------------------------
Image.fromarray(recon_std).save("recon_std.png")
Image.fromarray(recon_half).save("recon_half.png")
Image.fromarray(recon_double).save("recon_double.png")

# PSNR 和码率列表
psnrs = [psnr(img_gray,recon_half), psnr(img_gray,recon_std), psnr(img_gray,recon_double)]
rates = [rate_half, rate_std, rate_double]
labels = ['步长缩小', '标准', '步长放大']

# 绘制折线图
plt.figure()
plt.plot(rates, psnrs, marker='o')
for i, label in enumerate(labels):
    plt.text(rates[i], psnrs[i], label)
plt.xlabel('码率 (非零系数比例)')
plt.ylabel('PSNR (dB)')
plt.title('量化步长对码率与 PSNR 的影响')
plt.grid(True)
plt.savefig('rate_psnr.png')
plt.show()