# --- 1. Imports ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize as sk_resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time, os
from numba import jit


# --- 2. Load Image ---
USER_IMAGE_PATH = "/kaggle/input/image-super-resolution/dataset/train/high_res/11.png"
TARGET_SHAPE = (256, 256)
SCALE_FACTOR = 4

if not os.path.exists(USER_IMAGE_PATH):
    print(f"File not found: {USER_IMAGE_PATH}")
    original_gray = np.random.randint(0, 256, TARGET_SHAPE, dtype=np.uint8)
else:
    original_gray = cv2.imread(USER_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        print(f"Error loading: {USER_IMAGE_PATH}")
        original_gray = np.random.randint(0, 256, TARGET_SHAPE, dtype=np.uint8)

ground_truth = cv2.resize(original_gray, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
low_res_shape = (ground_truth.shape[0] // SCALE_FACTOR, ground_truth.shape[1] // SCALE_FACTOR)
target_dims_cv2 = (ground_truth.shape[1], ground_truth.shape[0])
low_res_input = sk_resize(ground_truth, low_res_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)

print(f"GT: {ground_truth.shape}, Low-res: {low_res_input.shape}")

# --- 3. Interpolation (1D) ---
@jit(nopython=True)
def weno_3_interpolate_1d(v):
    EPS = 1e-6
    v0, v1, v2 = v
    b0, b1 = (v1 - v0)**2, (v2 - v1)**2
    c0, c1 = 1/3, 2/3
    a0, a1 = c0 / (EPS + b0)**2, c1 / (EPS + b1)**2
    w0, w1 = a0 / (a0 + a1), a1 / (a0 + a1)
    p0, p1 = -0.5 * v0 + 1.5 * v1, 0.5 * v1 + 0.5 * v2
    return w0 * p0 + w1 * p1

@jit(nopython=True)
def linear_5_interpolate_1d(v):
    v0, v1, v2, v3, v4 = v
    return (1/30)*v0 - (13/60)*v1 + (47/60)*v2 + (9/20)*v3 - (1/20)*v4

@jit(nopython=True)
def weno_ao_interpolate_1d(v):
    return weno_3_interpolate_1d(v[1:4])

# --- 4. 2D Interpolation ---
@jit(nopython=True)
def interpolate_rows(img, func):
    pad = 2
    h, w = img.shape[0] - 2*pad, img.shape[1] - 2*pad
    out = np.zeros((h, w*2), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            out[r, c*2] = img[r+pad, c+pad]
            out[r, c*2+1] = func(img[r+pad, (c+pad-2):(c+pad+3)])
    return out

@jit(nopython=True)
def interpolate_cols(img, func):
    pad = 2
    h, w = img.shape[0] - 2*pad, img.shape[1]
    out = np.zeros((h*2, w), dtype=np.float64)
    for c in range(w):
        for r in range(h):
            out[r*2, c] = img[r+pad, c]
            out[r*2+1, c] = func(img[(r+pad-2):(r+pad+3), c])
    return out

def weno_ao_upscaler(img, scale):
    x = img.astype(np.float64)
    passes = int(np.log2(scale))
    if 2**passes != scale:
        raise ValueError("Only 2x, 4x, or 8x supported.")
    for i in range(passes):
        x = interpolate_cols(np.pad(interpolate_rows(np.pad(x, 2, mode='reflect'), weno_ao_interpolate_1d), ((2,2),(0,0)), mode='reflect'), weno_ao_interpolate_1d)
    return np.clip(x, 0, 255).astype(np.uint8)

# --- 5. Upscaling ---
print("Running upscalers...")
start = time.time()
img_nearest = cv2.resize(low_res_input, target_dims_cv2, interpolation=cv2.INTER_NEAREST)
img_bilinear = cv2.resize(low_res_input, target_dims_cv2, interpolation=cv2.INTER_LINEAR)
print(f"Standard done in {time.time()-start:.3f}s")

start = time.time()
img_weno = weno_ao_upscaler(low_res_input, SCALE_FACTOR)
print(f"WENO-AO done in {time.time()-start:.3f}s")

# --- 6. Metrics ---
results = {"Nearest": img_nearest, "Bilinear": img_bilinear, "WENO-AO": img_weno}
metrics = {}
print("\nMetrics:")
for k, img in results.items():
    p = psnr(ground_truth, img, data_range=255)
    s = ssim(ground_truth, img, data_range=255)
    metrics[k] = (p, s)
    print(f"{k}: PSNR={p:.2f}, SSIM={s:.4f}")

# --- 7. Plots ---
fig, ax = plt.subplots(2, 3, figsize=(18, 12))
plt.suptitle("Image Upscaling Comparison (4x)", fontsize=22, y=1.02)
a = ax.ravel()
a[0].imshow(ground_truth, cmap='gray'); a[0].set_title("GT"); a[0].axis('off')
a[1].imshow(low_res_input, cmap='gray'); a[1].set_title("Low-Res"); a[1].axis('off')
a[2].imshow(results["Nearest"], cmap='gray'); a[2].set_title(f"Nearest\nPSNR={metrics['Nearest'][0]:.2f}, SSIM={metrics['Nearest'][1]:.4f}"); a[2].axis('off')
a[3].imshow(results["Bilinear"], cmap='gray'); a[3].set_title(f"Bilinear\nPSNR={metrics['Bilinear'][0]:.2f}, SSIM={metrics['Bilinear'][1]:.4f}"); a[3].axis('off')
a[4].imshow(results["WENO-AO"], cmap='gray'); a[4].set_title(f"WENO-AO\nPSNR={metrics['WENO-AO'][0]:.2f}, SSIM={metrics['WENO-AO'][1]:.4f}"); a[4].axis('off')
a[5].axis('off')
plt.tight_layout()
plt.show()
