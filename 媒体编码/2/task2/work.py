import cv2
import numpy as np
import time

# ==========================
# 1. 生成 9 种 H.264 帧内预测模式
# ==========================
def intra_predict(block_size, mode, left=None, top=None):
    """
    简化的 H.264 预测模式生成
    mode: 0-8
    0: vertical, 1: horizontal, 2: DC, 3-8: 对角/其他
    left: 左边像素列
    top: 上边像素行
    """
    pred = np.zeros((block_size, block_size), dtype=np.uint8)

    if mode == 0:  # 垂直
        if top is not None:
            pred[:] = top
        else:
            pred[:] = 128
    elif mode == 1:  # 水平
        if left is not None:
            pred[:] = left[:, np.newaxis]
        else:
            pred[:] = 128
    elif mode == 2:  # DC
        values = []
        if top is not None:
            values.append(top)
        if left is not None:
            values.append(left)
        if values:
            pred[:] = int(np.mean(np.concatenate(values)))
        else:
            pred[:] = 128
    else:
        # 对角或其他模式简化为 DC + offset
        pred[:] = 128 + (mode-2)*5
        pred = np.clip(pred, 0, 255)

    return pred

# ==========================
# 2. 块残差和 PSNR
# ==========================
def block_sad(orig, pred):
    return np.sum(np.abs(orig - pred))

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255*255/mse)

# ==========================
# 3. 全遍历模式选择
# ==========================
def full_mode_decision(img, block_size=8):
    h, w = img.shape
    modes = np.zeros((h//block_size, w//block_size), dtype=int)
    residual = np.zeros_like(img)
    start = time.time()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            left = img[i:i+block_size, j-1] if j > 0 else None
            top = img[i-1, j:j+block_size] if i > 0 else None

            best_sad = float('inf')
            best_mode = 0
            for mode in range(9):
                pred = intra_predict(block_size, mode, left, top)
                s = block_sad(block, pred)
                if s < best_sad:
                    best_sad = s
                    best_mode = mode
                    best_pred = pred
            modes[i//block_size, j//block_size] = best_mode
            residual[i:i+block_size, j:j+block_size] = block - best_pred

    elapsed = time.time() - start
    return modes, residual, elapsed

# ==========================
# 4. 相邻块快速模式选择优化
# ==========================
def fast_mode_decision(img, block_size=8):
    h, w = img.shape
    modes = np.zeros((h//block_size, w//block_size), dtype=int)
    residual = np.zeros_like(img)
    start = time.time()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            left_mode = modes[i//block_size, (j-block_size)//block_size] if j > 0 else 2
            top_mode = modes[(i-block_size)//block_size, j//block_size] if i > 0 else 2

            # 候选模式：左/上相邻块最优模式 + DC
            candidate_modes = list(set([left_mode, top_mode, 2]))

            best_sad = float('inf')
            best_mode = candidate_modes[0]
            for mode in candidate_modes:
                left = img[i:i+block_size, j-1] if j > 0 else None
                top = img[i-1, j:j+block_size] if i > 0 else None
                pred = intra_predict(block_size, mode, left, top)
                s = block_sad(block, pred)
                if s < best_sad:
                    best_sad = s
                    best_mode = mode
                    best_pred = pred
            modes[i//block_size, j//block_size] = best_mode
            residual[i:i+block_size, j:j+block_size] = block - best_pred

    elapsed = time.time() - start
    return modes, residual, elapsed

# ==========================
# 5. 测试主程序
# ==========================
if __name__ == "__main__":
    img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)

    # 全遍历
    modes_full, res_full, t_full = full_mode_decision(img)
    # 快速优化
    modes_fast, res_fast, t_fast = fast_mode_decision(img)

    # PSNR
    psnr_val = psnr(res_full, res_fast)

    print(f"Full search time: {t_full:.3f}s")
    print(f"Fast search time: {t_fast:.3f}s")
    print(f"Time reduction: {(t_full-t_fast)/t_full*100:.2f}%")
    print(f"Residual PSNR difference: {psnr_val:.3f} dB")

    # 保存残差图
    cv2.imwrite("res_full.png", np.clip(res_full+128,0,255).astype(np.uint8))
    cv2.imwrite("res_fast.png", np.clip(res_fast+128,0,255).astype(np.uint8))
