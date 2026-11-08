import sys
sys.path.insert(0, r"c:\Users\ADARSH\Desktop\CV Hackathon")
from classical_isp import (
    ClassicalISP, subtract_black_level, gray_world_wb, white_patch_wb,
    apply_wb_gains, bilinear_demosaic, malvar_demosaic, bilateral_denoise,
    nlm_denoise, apply_ccm, compute_ccm_from_patches, DEFAULT_CCM,
    gamma_correction, srgb_gamma, reinhard_tone_map, filmic_tone_map,
    unsharp_mask, guided_filter_sharpen
)
import numpy as np

print("All imports OK")
isp = ClassicalISP()
print(f"ClassicalISP config keys: {list(isp.config.keys())}")
print(f"DEFAULT_CCM shape: {DEFAULT_CCM.shape}")

# Quick smoke test with synthetic Bayer data
raw = np.random.randint(64, 1024, (100, 100), dtype=np.uint16)
result = isp.process(raw, pattern='RGGB')
print(f"Output shape: {result.shape}, dtype: {result.dtype}, range: [{result.min()}, {result.max()}]")

steps = isp.process_steps(raw, pattern='RGGB')
print(f"Pipeline stages: {list(steps.keys())}")
print("All tests PASSED")
