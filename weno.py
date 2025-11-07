import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import time
import os
from numpy.lib.stride_tricks import sliding_window_view
import lpips
import torch
from brisque import BRISQUE

# ==============================================================================
# === 1. VECTORIZED WENO-AO(5,3) CORE FUNCTIONS
# ==============================================================================

def get_reconstructions_vec(v):
    """
    Vectorized calculation of reconstructions from a (N, 5) stencil array.
    """
    v0, v1, v2, v3, v4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4]
    
    # 3rd-order reconstructions
    p0 = (1/3)*v0 - (7/6)*v1 + (11/6)*v2
    p1 = (-1/6)*v1 + (5/6)*v2 + (1/3)*v3
    p2 = (1/3)*v2 + (5/6)*v3 - (1/6)*v4
    
    # 5th-order polynomial reconstruction
    ph = (1/30)*v0 - (13/60)*v1 + (47/60)*v2 + (27/60)*v3 - (3/60)*v4
    
    return p0, p1, p2, ph

def get_smoothness_indicators_vec(v):
    """
    Vectorized calculation of smoothness indicators from a (N, 5) stencil array.
    """
    v0, v1, v2, v3, v4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4]

    # Standard smoothness indicators (b0, b1, b2)
    b0 = (13/12)*(v0 - 2*v1 + v2)**2 + (1/4)*(v0 - 4*v1 + 3*v2)**2
    b1 = (13/12)*(v1 - 2*v2 + v3)**2 + (1/4)*(v1 - v3)**2
    b2 = (13/12)*(v2 - 2*v3 + v4)**2 + (1/4)*(3*v2 - 4*v3 + v4)**2
    
    # High-order smoothness indicator (bh)
    # Coefficients from Balsara et al. (2016)
    ux  = (1/120) * (11*v0 - 82*v1 + 82*v3 - 11*v4)
    ux2 = (1/56)  * (-3*v0 + 40*v1 - 74*v2 + 40*v3 - 3*v4)
    ux3 = (1/12)  * (-v0 + 2*v1 - 2*v3 + v4)
    ux4 = (1/24)  * (v0 - 4*v1 + 6*v2 - 4*v3 + v4)
    
    bh = (ux + ux3/10)**2 + (13/3)*(ux2 + 123*ux4/455)**2 + \
         (781/20)*(ux3**2) + (14211461/2275)*(ux4**2)
         
    return b0, b1, b2, bh

def weno_ao_5_3_reconstruction_vec(u):
    """
    Performs the full WENO-AO(5,3) adaptive order reconstruction on a 1D signal
    using NumPy vectorization for high speed.
    """
    # Pad the array to handle boundary stencils
    u_padded = np.pad(u, (2, 2), mode='edge')
    
    # --- Constants for the AO scheme ---
    epsilon = 1e-12 # avoid division by 0
    q = 2.0         
    #d0, d1, d2, dh = 0.1, 0.6, 0.3 ,1.0 
    d0, d1, d2, dh = 0.01125, 0.1275, 0.01125, 0.85
    # ------------------------------------

    # 1. Create all 5-point stencils at once
    # This creates a (N, 5) array, where N = len(u) - 1
    v = sliding_window_view(u_padded, 5)[:-1, :]
    
    # 2. Get all reconstruction polynomials (vectorized)
    p0, p1, p2, ph = get_reconstructions_vec(v)
    
    # 3. Get all smoothness indicators (vectorized)
    b0, b1, b2, bh = get_smoothness_indicators_vec(v)
    
    # 4. Calculate the "global" smoothness measure 'tau' (vectorized)
    tau = np.abs(bh - b0) + np.abs(bh - b1) + np.abs(bh - b2)
    
    # 5. Calculate the un-normalized adaptive weights (vectorized)
    # Reshape for broadcasting
    tau = tau[:, np.newaxis]
    bh = bh[:, np.newaxis]
    
    d_weights = np.array([d0, d1, d2])
    b_low = np.stack([b0, b1, b2], axis=1) # (N, 3) array
    p_low = np.stack([p0, p1, p2], axis=1) # (N, 3) array

    alpha_low = d_weights * (1.0 + (tau / (epsilon + b_low))**q)
    alpha_h = dh * (1.0 + (tau / (epsilon + bh))**q)
    
    # 6. Normalize the weights (vectorized)
    alpha_sum = np.sum(alpha_low, axis=1, keepdims=True) + alpha_h
    
    w_low = alpha_low / alpha_sum # (N, 3) array
    wh = alpha_h / alpha_sum      # (N, 1) array
    
    # 7. Compute the final blended reconstruction (vectorized)
    p_low_weighted_sum = np.sum(w_low * p_low, axis=1)
    d_low_weighted_sum = np.sum(d_weights * p_low, axis=1)
    
    recon_points = p_low_weighted_sum + (wh.flatten()/dh) * (ph - d_low_weighted_sum)
    
    return recon_points

# ==============================================================================
# === 2. 2D IMAGE APPLICATION FUNCTIONS
# ==============================================================================

def apply_weno_2d(image_channel):
    """
    Applies the 1D WENO reconstruction to a 2D image channel (grayscale).
    It first upscales horizontally, then vertically.
    
    NOTE: This function now calls the FAST `weno_ao_5_3_reconstruction_vec`
    """
    m, n = image_channel.shape
    
    # Target dimensions are (2M-1) x (2N-1)
    target_m, target_n = 2*m - 1, 2*n - 1
    
    # 1. Horizontal Upscaling
    # Create an intermediate image to hold the horizontally upscaled data
    interim_image = np.zeros((m, target_n))
    
    for i in range(m): # For each row
        row = image_channel[i, :]
        # Use the new vectorized function
        mid_points = weno_ao_5_3_reconstruction_vec(row)
        
        # "Weave" the original pixels and new mid-points
        interim_image[i, 0::2] = row
        interim_image[i, 1::2] = mid_points
        
    # 2. Vertical Upscaling
    # Create the final image
    final_image = np.zeros((target_m, target_n))
    
    for j in range(target_n): # For each column of the interim image
        col = interim_image[:, j]
        # Use the new vectorized function
        mid_points = weno_ao_5_3_reconstruction_vec(col)
        
        # "Weave" the original pixels and new mid-points
        final_image[0::2, j] = col
        final_image[1::2, j] = mid_points
        
    return final_image

def upscale_image_weno(lr_image_array):
    """
    Upscales a Low-Resolution (LR) image using WENO-AO(5,3).
    Handles both grayscale (2D) and color (3D) numpy arrays.
    """
    # Ensure data is float for calculations
    lr_image_array = lr_image_array.astype(np.float64)
    
    if lr_image_array.ndim == 3: # Color Image
        print("Processing 3-channel (RGB) image...")
        channels = []
        for i in range(3): # Process R, G, B channels separately
            print(f"  Upscaling channel {i+1}/3...")
            upscaled_channel = apply_weno_2d(lr_image_array[:, :, i])
            channels.append(upscaled_channel)
        
        # Stack channels back together
        upscaled_image = np.stack(channels, axis=-1)
        
    elif lr_image_array.ndim == 2: # Grayscale Image
        print("Processing 1-channel (grayscale) image...")
        upscaled_image = apply_weno_2d(lr_image_array)
        
    else:
        raise ValueError(f"Input image has {lr_image_array.ndim} dimensions. Expected 2 (grayscale) or 3 (color).")
        
    # Clip values to valid image range [0, 255]
    upscaled_image = np.clip(upscaled_image, 0, 255)
    
    return upscaled_image

# ==============================================================================
# === 3. METRIC CALCULATION FUNCTIONS
# ==============================================================================

def calculate_psnr(img_true, img_test):
    """
    Calculates PSNR between two images.
    Images are expected to be uint8 [0, 255].
    """
    return peak_signal_noise_ratio(img_true, img_test, data_range=255)

def calculate_ssim(img_true, img_test):
    """
    Calculates SSIM between two images.
    Handles both grayscale and multichannel (color).
    Dynamically adjusts 'win_size' for small images.
    """
    if img_true.ndim == 3:
        h, w, _ = img_true.shape
        channel_axis = 2
    elif img_true.ndim == 2:
        h, w = img_true.shape
        channel_axis = None
    else:
        raise ValueError(f"Input image has invalid dimensions: {img_true.shape}")
        
    min_dim = min(h, w)
    
    if min_dim < 7:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        if win_size < 3:
            print(f"Warning: Image dimensions ({h}x{w}) are too small for SSIM. Returning 0.")
            return 0.0
    else:
        win_size = 7
        
    return structural_similarity(
        img_true, 
        img_test, 
        data_range=255, 
        win_size=win_size,
        channel_axis=channel_axis
    )

def calculate_sharpness(image_uint8):
    """
    Measures sharpness using the variance of its Laplacian.
    Higher is sharper.
    """
    if image_uint8.ndim == 3:
        image_gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_uint8
    
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness
    
# Global variable to hold the LPIPS model so we only load it once
lpips_model = None

def calculate_lpips(img_true, img_test):
    """
    Calculates LPIPS (perceptual similarity) between two images.
    LOWER is better.
    Images must be torch.Tensor [C, H, W] in range [-1, 1].
    """
    global lpips_model
    if lpips_model is None:
        print("Loading LPIPS model (vgg)... This may take a moment.")
        # Use (net='vgg') for the best perceptual results
        # Set verbose=False to quiet the loading message
        lpips_model = lpips.LPIPS(net='vgg', verbose=False)
        print("LPIPS model loaded.")

    # 1. Convert numpy [0, 255] uint8 to torch [-1, 1] float32
    def to_tensor(img):
        img_float = img.astype(np.float32) / 255.0
        if img_float.ndim == 3:
            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)
        else: # Grayscale (H, W) to (1, H, W)
            img_tensor = torch.from_numpy(img_float).unsqueeze(0)
            img_tensor = img_tensor.repeat(3, 1, 1) # Replicate to 3 channels for VGG
            
        return (img_tensor * 2.0) - 1.0 # Normalize from [0, 1] to [-1, 1]

    tensor_true = to_tensor(img_true)
    tensor_test = to_tensor(img_test)

    # 2. Calculate LPIPS
    with torch.no_grad():
        score = lpips_model(tensor_true, tensor_test)
        
    return score.item()

brisque_model = None

# --- NEW METRIC: BRISQUE (Corrected) ---
def calculate_brisque(image_uint8):
    """
    Calculates the BRISQUE score (no-reference quality).
    LOWER is better.
    
    FIX: The brisque library's .score() method strictly requires 
    a 3-channel (color) image as input.
    """
    global brisque_model
    if brisque_model is None:
        print("Loading BRISQUE model...")
        brisque_model = BRISQUE(url=False) 
        print("BRISQUE model loaded.")

    # The BRISQUE library expects a 3-channel (color) image.
    
    if image_uint8.ndim == 2:
        # It's grayscale, convert it to 3-channel RGB
        image_color = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    else:
        image_color = image_uint8
    
    # The brisque package expects a float image in range [0, 255]
    image_color_float = image_color.astype(np.float32)

    try:
        # Pass the 3-channel float image to the score method
        score = brisque_model.score(image_color_float)
        return score
    except Exception as e:
        # Proactively catch the division-by-zero error we discussed
        if 'division by zero' in str(e).lower():
            print(f"Warning: BRISQUE calculation resulted in 'nan', likely due to a flat/empty image patch. Returning nan.")
            return np.nan
        else:
            print(f"Error calculating BRISQUE: {e}")
            return np.nan

# ==============================================================================
# === 4. MAIN EXECUTION SCRIPT
# ==============================================================================

def test_upsampling(LR_IMAGE_PATH,HR_IMAGE_PATH):
    try:
        # 2. Load Images
        lr_pil = Image.open(LR_IMAGE_PATH)
        hr_pil = Image.open(HR_IMAGE_PATH)
        
        lr_image_orig = np.array(lr_pil)
        hr_image_orig = np.array(hr_pil)
        
        print(f"Loaded LR image: {lr_image_orig.shape}")
        print(f"Loaded HR image: {hr_image_orig.shape}")
    
        # 3. Upscale the LR image using WENO-AO
        print("\nStarting WENO-AO(5,3) upscaling...")
        start_time = time.time()
        weno_upscaled_float = upscale_image_weno(lr_image_orig)
        weno_upscaled_uint8 = weno_upscaled_float.astype(np.uint8)
        end_time = time.time()
        print(f"WENO upscaling complete. Time taken: {end_time - start_time:.2f} seconds")
        base_filename = os.path.basename(LR_IMAGE_PATH)
        # Remove the original extension
        filename_no_ext, ext = os.path.splitext(base_filename)
        # Create a new descriptive filename
        output_filename = f"weno_upscaled_{filename_no_ext}.png"

        # Convert numpy array to PIL Image to save
        weno_pil_image = Image.fromarray(weno_upscaled_uint8)
        weno_pil_image.save(output_filename)
        print(f"WENOver upscaled image saved as: {output_filename}")
        # 4. Crop HR image for comparison
        # Output is (2M-1, 2N-1). Crop the HR (2M, 2N) to match.
        target_shape = weno_upscaled_uint8.shape
        hr_cropped_uint8 = hr_image_orig[:target_shape[0], :target_shape[1]]
        
        print(f"Upscaled image shape: {weno_upscaled_uint8.shape}")
        print(f"Cropped HR shape:     {hr_cropped_uint8.shape}")
    
        # 5. Run Comparison Upscalers
        print("\nRunning comparison upscalers (Bilinear, Bicubic)...")
        target_dims_pil = (target_shape[1], target_shape[0]) # PIL uses (width, height)
        
        bilinear_pil = lr_pil.resize(target_dims_pil, Image.Resampling.BILINEAR)
        bilinear_img = np.array(bilinear_pil)
        
        bicubic_pil = lr_pil.resize(target_dims_pil, Image.Resampling.BICUBIC)
        bicubic_img = np.array(bicubic_pil)
        print("Comparison upscalers complete.")
    
        # 6. Gather All Metrics
        methods = {
            "Bilinear": bilinear_img,
            "Bicubic": bicubic_img,
            "WENO-AO(5,3)": weno_upscaled_uint8
        }
        
        images_to_plot = {
            "Bilinear": bilinear_img,
            "Bicubic": bicubic_img,
            "WENO-AO(5,3)": weno_upscaled_uint8,
            "Ground Truth": hr_cropped_uint8
        }
        
        results = {}
        
        print("\n--- Quantitative Metrics Comparison ---")
        
        for name, img in images_to_plot.items():
            print(f"\nCalculating metrics for: {name}")
            
            # --- Handle metrics that require a comparison ---
            if name != "Ground Truth":
                if img.ndim != hr_cropped_uint8.ndim:
                    print(f"Warning: Mismatch in dimensions for {name}. Skipping comparison metrics.")
                    psnr_val = np.nan
                    ssim_val = np.nan
                    lpips_val = np.nan
                else:
                    psnr_val = calculate_psnr(hr_cropped_uint8, img)
                    ssim_val = calculate_ssim(hr_cropped_uint8, img)
                    lpips_val = calculate_lpips(hr_cropped_uint8, img) # NEW
            else:
                psnr_val = np.nan
                ssim_val = np.nan
                lpips_val = np.nan
    
            # --- Handle "no-reference" metrics ---
            sharp_val = calculate_sharpness(img)
            brisque_val = calculate_brisque(img) # NEW
            
            results[name] = {
                "PSNR (dB)": psnr_val,
                "SSIM (%)": ssim_val * 100 if not np.isnan(ssim_val) else np.nan,
                "LPIPS": lpips_val,
                "BRISQUE": brisque_val,
                "Sharpness": sharp_val
            }
            
            # --- Print to console ---
            print(f"  PSNR: {psnr_val:.2f} dB      (Higher is better)")
            print(f"  SSIM: {results[name]['SSIM (%)']:.2f} %     (Higher is better)")
            print(f"  LPIPS: {lpips_val:.4f}        (LOWER is better)")
            print(f"  BRISQUE: {brisque_val:.2f}    (LOWER is better)")
            print(f"  Sharpness: {sharp_val:.2f}   (Higher is sharper)")
    
    
        # 7. Display Results
        print("\n--- Generating Visual Comparison Plots ---")
        
        is_gray = hr_cropped_uint8.ndim == 2
        cmap = 'gray' if is_gray else None
        
        # --- Plot 1: Full Image Comparison ---
        fig_full, axes_full = plt.subplots(1, 4, figsize=(24, 8))
        
        for ax, (name, img) in zip(axes_full, images_to_plot.items()):
            ax.imshow(img, cmap=cmap)
            
            # Get metrics text
            psnr = results[name]["PSNR (dB)"]
            ssim = results[name]["SSIM (%)"]
            lpips = results[name]["LPIPS"]
            brisque = results[name]["BRISQUE"]
            
            if name == "Ground Truth":
                title_text = f"{name}\nBRISQUE: {brisque:.2f}"
            else:
                title_text = (
                    f"{name}\n"
                    f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.2f} %\n"
                    f"LPIPS: {lpips:.4f} | BRISQUE: {brisque:.2f}"
                )
            ax.set_title(title_text, fontsize=11)
            ax.axis('off')
    
        fig_full.suptitle("Full Image Comparison (WENO-AO vs. Standard)", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig('weno_comparison_full.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # --- Plot 2: Zoomed-In Artifact Analysis ---
        print("--- Generating Zoomed-In Crop Plot ---")
        
        h, w = hr_cropped_uint8.shape[:2]
        y_start, y_end = int(h * 0.4), int(h * 0.6)
        x_start, x_end = int(w * 0.4), int(w * 0.6)
        
        if y_start == y_end: y_end += 1
        if x_start == x_end: x_end += 1
            
        crop_slice = (slice(y_start, y_end), slice(x_start, x_end))
        print(f"Using crop region: Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}]")
    
        fig_zoom, axes_zoom = plt.subplots(1, 4, figsize=(20, 6))
    
        for ax, (name, img) in zip(axes_zoom, images_to_plot.items()):
            img_crop = img[crop_slice]
            ax.imshow(img_crop, cmap=cmap, interpolation='nearest') # 'nearest' shows raw pixels
            ax.set_title(name, fontsize=14)
            ax.axis('off')
    
        fig_zoom.suptitle("Zoomed-In Crop for Artifact Analysis (Center 20% Patch)", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig('weno_comparison_zoom.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    except FileNotFoundError:
        print(f"--- 🚫 ERROR ---")
        print(f"Image files not found. Please update the variables:")
        print(f"LR_IMAGE_PATH = \"{LR_IMAGE_PATH}\"")
        print(f"HR_IMAGE_PATH = \"{HR_IMAGE_PATH}\"")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

LR_IMAGE_PATH = "/kaggle/input/urban100/Urban 100/X2 Urban100/X2/LOW X2 Urban/img_001_SRF_2_LR.png"  # e.g., "images/road_lr_256.png"
HR_IMAGE_PATH = "/kaggle/input/urban100/Urban 100/X2 Urban100/X2/HIGH X2 Urban/img_001_SRF_2_HR.png"  # e.g., "images/road_hr_512.png"
test_upsampling(LR_IMAGE_PATH,HR_IMAGE_PATH)
