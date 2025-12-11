# test_generator.py
# Generates CIF (352x288) YUV420p sequence with a moving square for testing.
from PIL import Image, ImageDraw
import numpy as np

W, H = 352, 288
FRAMES = 150
OUT = "test_cif.yuv"

def rgb_to_yuv420p(img):
    r = img[:,:,0].astype(np.int32)
    g = img[:,:,1].astype(np.int32)
    b = img[:,:,2].astype(np.int32)
    y = (  66*r + 129*g +  25*b + 128) >> 8
    y = y + 16
    u = ((-38*r -  74*g + 112*b + 128) >> 8) + 128
    v = ((112*r -  94*g -  18*b + 128) >> 8) + 128
    y = np.clip(y,0,255).astype(np.uint8)
    u = np.clip(u,0,255).astype(np.uint8)
    v = np.clip(v,0,255).astype(np.uint8)
    u_sub = u.reshape(H//2,2,W//2,2).mean(axis=(1,3)).astype(np.uint8)
    v_sub = v.reshape(H//2,2,W//2,2).mean(axis=(1,3)).astype(np.uint8)
    return y, u_sub, v_sub

def make_frame(t):
    img = Image.new("RGB",(W,H),(64,64,64))
    draw = ImageDraw.Draw(img)
    size = 48
    x = int((W - size) * ((t % 60) / 59.0))
    y = int((H - size) * ((t % 40) / 39.0))
    draw.rectangle([x,y,x+size,y+size], fill=(200,40,40))
    return np.array(img)

def main():
    with open(OUT,"wb") as f:
        for i in range(FRAMES):
            img = make_frame(i)
            y,u,v = rgb_to_yuv420p(img)
            f.write(y.tobytes())
            f.write(u.tobytes())
            f.write(v.tobytes())
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
