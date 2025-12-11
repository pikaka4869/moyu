# entropy_package.py
# 完整熵编码仿真包：Huffman / Arithmetic / CAVLC，包含上下文建模优化 & RD 曲线
# 运行： python entropy_package.py
# 依赖： numpy scipy pillow matplotlib scikit-learn python-docx (可选)

import os, time, math
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import heapq
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ------------------------------
# Utility: DCT / IDCT / Quantize
# ------------------------------
STD_QTABLE = np.array([
 [16,11,10,16,24,40,51,61],
 [12,12,14,19,26,58,60,55],
 [14,13,16,24,40,57,69,56],
 [14,17,22,29,51,87,80,62],
 [18,22,37,56,68,109,103,77],
 [24,35,55,64,81,104,113,92],
 [49,64,78,87,103,121,120,101],
 [72,92,95,98,112,100,103,99]
], dtype=np.float32)

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(coeff):
    return idct(idct(coeff.T, norm='ortho').T, norm='ortho')

def quantize_block(block, qtable):
    D = dct2(block - 128.0)
    Q = np.round(D / qtable).astype(int)
    return Q

def dequantize_block(Q, qtable):
    Drec = Q.astype(float) * qtable
    block = idct2(Drec) + 128.0
    return np.clip(block, 0, 255).astype(np.uint8)

# ------------------------------
# Image -> blocks (8x8) helper
# ------------------------------
def image_to_blocks(img_array):
    h,w = img_array.shape
    blocks = []
    coords = []
    for i in range(0,h,8):
        for j in range(0,w,8):
            blk = img_array[i:i+8, j:j+8]
            if blk.shape != (8,8): continue
            blocks.append(blk.copy())
            coords.append((i,j))
    return np.array(blocks), coords

def blocks_to_image(blocks, coords, shape):
    img = np.zeros(shape, dtype=np.uint8)
    for b,(i,j) in zip(blocks, coords):
        img[i:i+8, j:j+8] = b
    return img

# ------------------------------
# Zig-zag scan (8x8) -> 1D
# ------------------------------
ZIGZAG_INDEX = [
(0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
(2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
(1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
(3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
(4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
(3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
(7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
(6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]

def zigzag_flat(block):
    return np.array([ block[i,j] for (i,j) in ZIGZAG_INDEX ], dtype=int)

# ------------------------------
# Huffman coding (static + dynamic)
# ------------------------------
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        if self.freq == other.freq:
            return (self.symbol or 0) < (other.symbol or 0)
        return self.freq < other.freq

def build_huffman_tree(freq_map):
    heap = [HuffmanNode(s,f) for s,f in freq_map.items()]
    if len(heap)==0:
        return None
    heapq.heapify(heap)
    if len(heap)==1:
        # special-case
        node = heapq.heappop(heap)
        parent = HuffmanNode(freq=node.freq)
        parent.left = node
        parent.right = HuffmanNode(symbol=None, freq=0)
        return parent
    while len(heap)>1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        p = HuffmanNode(freq=n1.freq + n2.freq)
        p.left = n1; p.right = n2
        heapq.heappush(heap, p)
    return heap[0]

def generate_code_table(node):
    table = {}
    def dfs(n, prefix):
        if n is None: return
        if n.symbol is not None:
            table[n.symbol] = prefix if prefix!="" else "0"
            return
        dfs(n.left, prefix+"0")
        dfs(n.right, prefix+"1")
    dfs(node, "")
    return table

def serialize_freq_map(freq_map):
    # serialize as: n_entries (2 bytes) then for each entry: symbol (4 bytes signed) + freq (8 bytes)
    # using simple text encoding for portability
    items = []
    for s,f in freq_map.items():
        items.append(f"{int(s)}:{int(f)}")
    return ",".join(items).encode('utf-8')

def deserialize_freq_map(byte_data):
    txt = byte_data.decode('utf-8')
    freq_map = {}
    if txt.strip()=="":
        return freq_map
    for token in txt.split(","):
        s,f = token.split(":")
        freq_map[int(s)] = int(f)
    return freq_map

# encode with static global table
def huffman_static_encode(symbols):
    freq = Counter(symbols.tolist())
    root = build_huffman_tree(freq)
    table = generate_code_table(root)
    bitstr = ''.join(table[s] for s in symbols)
    meta = serialize_freq_map(freq)  # decoder can rebuild tree
    return meta + b"\n" + bitstr.encode('utf-8'), table, root

def huffman_static_decode(stream_bytes):
    meta, bitbytes = stream_bytes.split(b"\n",1)
    freq = deserialize_freq_map(meta)
    root = build_huffman_tree(freq)
    table = generate_code_table(root)
    rev = {v:k for k,v in table.items()}
    bitstr = bitbytes.decode('utf-8')
    out = []
    cur = ""
    for b in bitstr:
        cur += b
        if cur in rev:
            out.append(rev[cur])
            cur = ""
    return np.array(out, dtype=int), table, root

# dynamic per-block: encode each block separately, prefix each block with frequency map size (text)
def huffman_dynamic_encode_per_block(blocks_flat):
    # blocks_flat: list of 1D numpy arrays (each block zigzag flattened)
    chunks = []
    total_bits = 0
    tables = []
    for blk in blocks_flat:
        freq = Counter(blk.tolist())
        meta = serialize_freq_map(freq)
        root = build_huffman_tree(freq)
        table = generate_code_table(root)
        bitstr = ''.join(table[s] for s in blk)
        chunk = meta + b"\n" + bitstr.encode('utf-8')
        chunks.append(chunk)
        total_bits += len(bitstr)
        tables.append(table)
    stream = b"||B||".join(chunks)  # block separator
    return stream, tables, total_bits

def huffman_dynamic_decode_per_block(stream):
    chunks = stream.split(b"||B||")
    out = []
    tables = []
    for ch in chunks:
        if len(ch.strip())==0: continue
        meta, bitbytes = ch.split(b"\n",1)
        freq = deserialize_freq_map(meta)
        root = build_huffman_tree(freq)
        table = generate_code_table(root)
        rev = {v:k for k,v in table.items()}
        bitstr = bitbytes.decode('utf-8')
        cur=""
        blkout=[]
        for b in bitstr:
            cur += b
            if cur in rev:
                blkout.append(int(rev[cur]))
                cur = ""
        out.append(np.array(blkout,dtype=int))
        tables.append(table)
    return out, tables

# ------------------------------
# Simple Arithmetic coder (integer implementation)
# Classic range coder with 32-bit range - educational version
# ------------------------------
class ArithmeticCoder:
    def __init__(self, precision=32):
        self.PREC = precision
        self.MAX_RANGE = 1 << self.PREC
        self.HALF = self.MAX_RANGE >> 1
        self.QUARTER = self.HALF >> 1

    def encode(self, symbols, freq_map):
        # freq_map: dict symbol->freq (non-zero), used as static model (cumulative)
        # Build cumulative frequencies
        total = sum(freq_map.values())
        syms = sorted(freq_map.keys())
        cum = {}
        s=0
        for sym in syms:
            cum[sym] = (s, s+freq_map[sym])
            s += freq_map[sym]
        low = 0
        high = self.MAX_RANGE - 1
        out_bits = []
        underflow = 0

        def output_bit(b):
            out_bits.append(str(b))

        for sym in symbols:
            if sym not in cum:
                # handle unseen symbol (shouldn't happen if model built from data)
                lo_count, hi_count = 0,1
            else:
                lo_count, hi_count = cum[sym]
            range_size = high - low + 1
            high = low + (range_size * hi_count)//total -1
            low  = low + (range_size * lo_count)//total
            # renormalize
            while True:
                if high < self.HALF:
                    output_bit(0)
                    for _ in range(underflow):
                        output_bit(1)
                    underflow = 0
                    low = low << 1
                    high = (high << 1) + 1
                elif low >= self.HALF:
                    output_bit(1)
                    for _ in range(underflow):
                        output_bit(0)
                    underflow = 0
                    low = (low - self.HALF) << 1
                    high = ((high - self.HALF) << 1) + 1
                elif low >= self.QUARTER and high < 3*self.QUARTER:
                    underflow += 1
                    low = (low - self.QUARTER) << 1
                    high = ((high - self.QUARTER) << 1) + 1
                else:
                    break
        # flush
        underflow += 1
        if low < self.QUARTER:
            output_bit(0)
            for _ in range(underflow):
                output_bit(1)
        else:
            output_bit(1)
            for _ in range(underflow):
                output_bit(0)
        bitstr = ''.join(out_bits)
        # For decoding, we must return also the model (freq_map) to allow reconstruction.
        meta = serialize_freq_map(freq_map)
        return meta + b"\n" + bitstr.encode('utf-8')

    def decode(self, stream, nsymbols):
        meta, bitbytes = stream.split(b"\n",1)
        freq_map = deserialize_freq_map(meta)
        total = sum(freq_map.values())
        syms = sorted(freq_map.keys())
        cum = []
        s=0
        for sym in syms:
            cum.append((sym, s, s+freq_map[sym]))
            s += freq_map[sym]
        bitstr = bitbytes.decode('utf-8')
        # prepare bit generator
        bits_iter = iter(bitstr)
        def read_bit():
            try:
                return int(next(bits_iter))
            except StopIteration:
                return 0
        # Initialize value
        value = 0
        for _ in range(self.PREC):
            value = (value << 1) + read_bit()
        low = 0
        high = self.MAX_RANGE - 1
        out = []
        for _ in range(nsymbols):
            range_size = high - low + 1
            scaled = ((value - low + 1) * total - 1) // range_size
            # find symbol where cum_lo <= scaled < cum_hi
            sym = None
            for (sval, lo_c, hi_c) in cum:
                if lo_c <= scaled < hi_c:
                    sym = sval
                    lo_count, hi_count = lo_c, hi_c
                    break
            if sym is None:
                sym = cum[-1][0]
                lo_count, hi_count = cum[-1][1], cum[-1][2]
            out.append(int(sym))
            high = low + (range_size * hi_count)//total -1
            low  = low + (range_size * lo_count)//total
            # renormalize
            while True:
                if high < self.HALF:
                    pass
                elif low >= self.HALF:
                    value -= self.HALF
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.QUARTER and high < 3*self.QUARTER:
                    value -= self.QUARTER
                    low -= self.QUARTER
                    high -= self.QUARTER
                else:
                    break
                low = (low << 1) & (self.MAX_RANGE-1)
                high = ((high << 1) & (self.MAX_RANGE-1)) | 1
                value = ((value << 1) & (self.MAX_RANGE-1)) | read_bit()
        return np.array(out, dtype=int), deserialize_freq_map(meta)

# ------------------------------
# Simplified CAVLC
# ------------------------------
def cavlc_encode_blocks(blocks):
    # blocks: list of 8x8 quantized int arrays
    bits = []
    for b in blocks:
        seq = zigzag_flat(b)
        run = 0
        for v in seq:
            if v == 0:
                run += 1
            else:
                # encode run as unary: run zeros then '1', then level as sign + 8-bit abs
                bits.append('0'*run + '1')
                level = int(v)
                sign = '1' if level < 0 else '0'
                bits.append(sign + format(abs(level)&0xff,'08b'))
                run = 0
        bits.append('1')  # EOB marker: single '1'
    return ''.join(bits)

# ------------------------------
# Context modeling optimization (示例)
#  - 用块能量或块中非零率作为特征做 k-means 聚类（上下文分组）
#  - 在每个上下文里单独训练 H/Arithmetic 模型，再编码以比较码率
# ------------------------------
def compute_block_features(blocks):
    feats = []
    for b in blocks:
        seq = zigzag_flat(b)
        nonzero = np.count_nonzero(seq)
        energy = np.sum(seq.astype(float)**2)
        feats.append([nonzero, energy])
    return np.array(feats)

def context_cluster_and_encode(blocks, nqpoints=3):
    feats = compute_block_features(blocks)
    if not SKLEARN_AVAILABLE:
        # fallback: simple threshold grouping by nonzero count quantiles
        nz = feats[:,0]
        thresholds = np.percentile(nz, [33,66])
        groups = np.digitize(nz, thresholds)
    else:
        kmeans = KMeans(n_clusters=nqpoints, random_state=0).fit(feats)
        groups = kmeans.labels_
    # Build per-group frequency maps and encode using arithmetic coder
    groups_indices = defaultdict(list)
    for idx,g in enumerate(groups):
        groups_indices[g].append(idx)
    coder = ArithmeticCoder()
    total_bits = 0.0
    total_symbols = 0
    t0 = time.time()
    for g, idxs in groups_indices.items():
        # collect all symbols in this context (flattened)
        syms = []
        for i in idxs:
            syms.extend( zigzag_flat(blocks[i]).tolist() )
        freq = Counter(syms)
        meta = serialize_freq_map(freq)
        encoded = coder.encode(np.array(syms,dtype=int), freq)
        # estimate bits length from returned bitstring
        parts = encoded.split(b"\n",1)
        bitlen = 0
        if len(parts)>1:
            bitlen = len(parts[1])  # number of bytes -> chars = bits since we encoded '0'/'1' text
        total_bits += bitlen
        total_symbols += len(syms)
    t1 = time.time()
    rate = total_bits / total_symbols if total_symbols>0 else float('inf')
    elapsed = t1 - t0
    return rate, elapsed, groups

# ------------------------------
# RD Curve sweep
# ------------------------------
def run_rd_sweep(img_path="lena.png", qscales=[0.5,1.0,1.5,2.0,3.0]):
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    blocks, coords = image_to_blocks(arr)
    results = {'Huffman_static':[], 'Huffman_dynamic':[], 'CAVLC':[], 'Arithmetic':[], 'ContextOpt':[]}
    coder = ArithmeticCoder()
    log_lines = []
    for qs in qscales:
        qtable = np.clip(STD_QTABLE * qs, 1, 255)
        # quantize
        qblocks = []
        for blk in blocks:
            Q = quantize_block(blk, qtable)  # returns integers
            qblocks.append(Q)
        qblocks = np.array(qblocks)
        # rebuild symbols flattened for static
        syms = np.concatenate([zigzag_flat(b) for b in qblocks])
        # static Huffman
        t0=time.time()
        stream_static, table_static, root = huffman_static_encode(syms)
        t1=time.time()
        rate_static = len(stream_static.split(b"\n",1)[1]) / syms.size  # bits per symbol (approx)
        # static decode check
        decoded,_,_ = huffman_static_decode(stream_static)
        ok_static = np.array_equal(decoded, syms)
        # dynamic Huffman per-block
        blocks_flat = [zigzag_flat(b) for b in qblocks]
        t0=time.time()
        stream_dyn, tables_dyn, total_bits_dyn = huffman_dynamic_encode_per_block(blocks_flat)
        t1=time.time()
        rate_dyn = total_bits_dyn / syms.size
        # cavlc
        t0=time.time()
        stream_cavlc = cavlc_encode_blocks(qblocks)
        t1=time.time()
        rate_cavlc = len(stream_cavlc) / syms.size
        # arithmetic (global static frequency)
        freq = Counter(syms.tolist())
        t0=time.time()
        stream_arith = coder.encode(syms, freq)
        t1=time.time()
        # estimate arithmetic bits
        bit_part = stream_arith.split(b"\n",1)[1] if b"\n" in stream_arith else b''
        rate_arith = len(bit_part) / syms.size
        # context optimization
        rate_ctx, t_ctx, groups = context_cluster_and_encode(qblocks, nqpoints=3)
        # reconstruct image with dequantize using qtable and compute PSNR
        recon_blocks = [ dequantize_block(b, qtable) for b in qblocks ]
        recon_img = blocks_to_image(recon_blocks, coords, arr.shape)
        mse = np.mean((arr - recon_img.astype(float))**2)
        psnr = 10 * math.log10((255.0**2) / mse) if mse>0 else 999.0
        # log
        results['Huffman_static'].append((rate_static, psnr))
        results['Huffman_dynamic'].append((rate_dyn, psnr))
        results['CAVLC'].append((rate_cavlc, psnr))
        results['Arithmetic'].append((rate_arith, psnr))
        results['ContextOpt'].append((rate_ctx, psnr))
        log_lines.append(f"qs={qs:.2f}: static={rate_static:.4f}, dyn={rate_dyn:.4f}, cavlc={rate_cavlc:.4f}, arith={rate_arith:.4f}, ctx={rate_ctx:.4f}, PSNR={psnr:.2f}")
        print(log_lines[-1])
    # plotting RD curves
    plt.figure(figsize=(8,5))
    markers = {'Huffman_static':'o','Huffman_dynamic':'s','CAVLC':'^','Arithmetic':'x','ContextOpt':'*'}
    for k,v in results.items():
        rates = [p[0] for p in v]
        ps = [p[1] for p in v]
        plt.plot(rates, ps, marker=markers[k], label=k)
        # mark best (max PSNR for similar rate) — we'll annotate max psnr point
        best_idx = int(np.argmin(rates))  # lowest rate point for each alg (example)
        plt.scatter([rates[best_idx]],[ps[best_idx]], s=80)
    plt.xlabel("Rate (bits per quantized coefficient, approx)")
    plt.ylabel("PSNR (dB)")
    plt.title("RD Curves: different entropy coding")
    plt.legend()
    plt.grid(True)
    plt.savefig("rate_psnr.png", dpi=200)
    plt.close()
    # save logs
    with open("entropy_simulation_log.txt","w",encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    # context optimization report
    with open("context_optimization_report.txt","w",encoding="utf-8") as f:
        f.write("Context optimization summary (k-means groups if available):\n")
        f.write("\n".join(log_lines))
    return results

# ------------------------------
# Main
# ------------------------------
def main():
    IMG = "lena.png"
    if not os.path.exists(IMG):
        print("请把 lena.png 放在当前目录后再运行")
        return
    qscales = [0.5, 1.0, 1.5, 2.0, 3.0]
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not installed; context clustering will use simple thresholds.")
    results = run_rd_sweep(IMG, qscales=qscales)
    print("完成：生成 rate_psnr.png, entropy_simulation_log.txt, context_optimization_report.txt")
    print("检查生成的 RD 曲线与报告，确认上下文优化是否达到 ≥10% 的码率降低（报告中有数据）")

if __name__ == "__main__":
    main()
