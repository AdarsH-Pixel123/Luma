"""Profile and compare model efficiency metrics."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_isp.mobile_unet import MobileUNet
from neural_isp.pynet_lite import PyNETLite
from optimization.profiler import full_profile, profile_comparison
from optimization.quantize import convert_to_fp16, compare_model_sizes


def main():
    print("=" * 70)
    print("MOBILE ISP MODEL PROFILING & EFFICIENCY BENCHMARK")
    print("=" * 70)

    models = {
        'MobileUNet (FP32)': MobileUNet(),
        'PyNETLite (FP32)': PyNETLite(),
    }

    results = profile_comparison(models, input_shape=(1, 4, 128, 128))

    for name, metrics in results.items():
        print(f"\nModel: {name}")
        print("-" * 50)
        for k, v in metrics.items():
            print(f"  {k:<20}: {v}")

    # Precision and Model Size Comparison
    print("\n" + "=" * 70)
    print("STORAGE FOOTPRINT: FP32 vs FP16 PRECISION")
    print("=" * 70)

    precision_variants = {
        'MobileUNet FP32': MobileUNet(),
        'MobileUNet FP16': convert_to_fp16(MobileUNet()),
        'PyNETLite FP32': PyNETLite(),
        'PyNETLite FP16': convert_to_fp16(PyNETLite()),
    }

    sizes = compare_model_sizes(precision_variants)
    for name, size in sizes.items():
        print(f"  {name:<25}: {size:>6.2f} MB")
    print("=" * 70)


if __name__ == '__main__':
    main()
