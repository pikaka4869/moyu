import numpy as np
import cv2
import time

# ==========================
# 1. 读取 Y4M 文件
# ==========================
def read_y4m(filename, num_frames=2):
    """
    读取 Y4M 文件，返回 list，每个元素是 (Y, U, V)
    """
    frames = []
    with open(filename, 'rb') as f:
        # 读取文件头
        header = f.readline().decode('ascii').strip()
        # 解析分辨率
        width = height = None
        for part in header.split():
            if part.startswith('W'):
                width = int(part[1:])
            elif part.startswith('H'):
                height = int(part[1:])
        if width is None or height is None:
            raise ValueError("无法解析 Y4M 分辨率")
        
        frame_idx = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(b'FRAME'):
                # 读取 Y
                y = np.frombuffer(f.read(width*height), dtype=np.uint8).reshape((height, width))
                # 读取 U
                u = np.frombuffer(f.read((width*height)//4), dtype=np.uint8).reshape((height//2, width//2))
                # 读取 V
                v = np.frombuffer(f.read((width*height)//4), dtype=np.uint8).reshape((height//2, width//2))
                frames.append((y, u, v))
                frame_idx += 1
                if frame_idx >= num_frames:
                    break
    return frames, width, height

# ==========================
# 2. SAD 代价函数
# ==========================
def sad(block1, block2):
    return np.sum(np.abs(block1 - block2))

# ==========================
# 3. 全搜索 FS
# ==========================
def full_search(prev, curr, block_size=16, search_range=16):
    h, w = curr.shape
    mv = np.zeros((h//block_size, w//block_size, 2), dtype=int)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            min_sad = float('inf')
            best_dx, best_dy = 0, 0
            block = curr[i:i+block_size, j:j+block_size]
            for dx in range(-search_range, search_range+1):
                for dy in range(-search_range, search_range+1):
                    ref_x = j + dx
                    ref_y = i + dy
                    if 0 <= ref_x <= w-block_size and 0 <= ref_y <= h-block_size:
                        ref_block = prev[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                        s = sad(block, ref_block)
                        if s < min_sad:
                            min_sad = s
                            best_dx, best_dy = dx, dy
            mv[i//block_size, j//block_size] = [best_dx, best_dy]
    return mv

# ==========================
# 4. 三步搜索 TSS
# ==========================
def three_step_search(prev, curr, block_size=16, search_range=16):
    h, w = curr.shape
    mv = np.zeros((h//block_size, w//block_size, 2), dtype=int)
    step_init = max(1, search_range // 2)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = curr[i:i+block_size, j:j+block_size]
            center_x, center_y = j, i
            step = step_init
            best_dx, best_dy = 0, 0

            while step >= 1:
                min_sad = float('inf')
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        ref_x = center_x + best_dx + dx
                        ref_y = center_y + best_dy + dy
                        if 0 <= ref_x <= w-block_size and 0 <= ref_y <= h-block_size:
                            ref_block = prev[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                            s = sad(block, ref_block)
                            if s < min_sad:
                                min_sad = s
                                best_dx += dx
                                best_dy += dy
                step //= 2
            mv[i//block_size, j//block_size] = [best_dx, best_dy]
    return mv

# ==========================
# 5. 绘制运动向量图
# ==========================
def draw_motion_vectors(frame, mv, block_size=16):
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for i in range(mv.shape[0]):
        for j in range(mv.shape[1]):
            start = (j*block_size + block_size//2, i*block_size + block_size//2)
            dx, dy = mv[i,j]
            end = (start[0]+dx, start[1]+dy)
            cv2.arrowedLine(img, start, end, (0,0,255), 1, tipLength=0.3)
    return img

# ==========================
# 6. 残差与 PSNR
# ==========================
def compute_residual(prev, mv, block_size=16):
    h, w = prev.shape
    residual = np.zeros_like(prev)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dx, dy = mv[i//block_size, j//block_size]
            ref_x = j + dx
            ref_y = i + dy
            if 0 <= ref_x <= w-block_size and 0 <= ref_y <= h-block_size:
                ref_block = prev[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                residual[i:i+block_size, j:j+block_size] = prev[i:i+block_size, j:j+block_size] - ref_block
    return residual

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255*255/mse)

# ==========================
# 7. 主程序
# ==========================
if __name__ == "__main__":
    filename = "foreman_cif.y4m"
    frames, width, height = read_y4m(filename, num_frames=2)
    y0, y1 = frames[0][0], frames[1][0]

    print(f"Video resolution: {width}x{height}")

    # 全搜索 FS
    start = time.time()
    mv_fs = full_search(y0, y1)
    fs_time = time.time() - start
    print("Full Search Time: {:.2f}s".format(fs_time))

    # 三步搜索 TSS
    start = time.time()
    mv_tss = three_step_search(y0, y1)
    tss_time = time.time() - start
    print("Three-Step Search Time: {:.2f}s".format(tss_time))

    # 加速比
    speedup = (fs_time - tss_time)/fs_time*100
    print("TSS Speedup: {:.2f}%".format(speedup))

    # 运动向量可视化
    img_fs = draw_motion_vectors(y1, mv_fs)
    img_tss = draw_motion_vectors(y1, mv_tss)
    cv2.imwrite("mv_fs.png", img_fs)
    cv2.imwrite("mv_tss.png", img_tss)

    # 残差与 PSNR
    residual_fs = compute_residual(y0, mv_fs)
    residual_tss = compute_residual(y0, mv_tss)
    psnr_value = psnr(residual_fs, residual_tss)
    print("PSNR (FS vs TSS residuals): {:.2f} dB".format(psnr_value))

    # 保存残差图
    cv2.imwrite("residual_fs.png", np.clip(residual_fs+128,0,255).astype(np.uint8))
    cv2.imwrite("residual_tss.png", np.clip(residual_tss+128,0,255).astype(np.uint8))
