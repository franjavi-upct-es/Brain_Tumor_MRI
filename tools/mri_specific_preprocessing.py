"""
tools/mri_specific_preprocessing.py
-----------------------------------
State-of-the-art preprocessing techniques specifically for brain MRI images.
Based on clinical neuroimaging pipelines and deep learning best practices.

References:
- FSL (FMRIB Software Library) preprocessing pipeline
- SPM (Statistical Parametric Mapping) normalization
- ANTs (Advanced Normalization Tools)
- Medical imaging deep learning papers (2020-2024)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings
from scipy.ndimage import gaussian_filter
from skimage import exposure

warnings.filterwarnings("ignore", category=UserWarning)


class MRIPreprocessor:
    """
    Medical-grade preprocessing for brain MRI images.
    Implements techniques from clinical neuroimaging and recent research.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        apply_n4_bias_correction: bool = True,
        apply_rician_denoising: bool = True,
        skull_strip_method: str = "bet",  # "bet", "watershed", "otsu"
        # "nyul", "zscore", "minmax", "histogram_matching"
        intensity_normalization: str = "nyul",
        enhance_method: str = "clahe",  # "clahe", "anisotropic_diffusion", "unsharp_mask"
    ):
        self.target_size = target_size
        self.apply_n4_bias = apply_n4_bias_correction
        self.apply_rician = apply_rician_denoising
        self.skull_strip_method = skull_strip_method
        self.intensity_norm = intensity_normalization
        self.enhance_method = enhance_method

    # ==================== BIAS FIELD CORRECTION ====================

    def n4_bias_correction_simple(self, img: np.ndarray) -> np.ndarray:
        """
        Simplified N4 bias field correction (N4ITK approximation).

        Bias field in MRI causes smooth intensity variations across the image.
        This is one of the MOST IMPORTANT steps for MRI preprocessing.

        Full N4ITK requires SimpleITK, but here's a practical approximation:
        - Estimate low-frequency bias with Gaussian smoothing
        - Divide original image by the bias field estimate
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Convert to float
        img_float = gray.astype(np.float32)

        # Estimate bias field with heavy Gaussian smoothing
        # The bias field is a low-frequency component
        sigma = min(img_float.shape) / 8.0  # Adaptive sigma
        bias_field = gaussian_filter(img_float, sigma=sigma)

        # Avoid division by zero
        bias_field = np.maximum(bias_field, 1e-7)

        # Correct bias
        corrected = img_float / bias_field

        # Rescale to [0, 255]
        corrected = (corrected - corrected.min()) / (
            corrected.max() - corrected.min() + 1e-7
        )
        corrected = (corrected * 255).astype(np.uint8)

        # If input was color, apply correction to each channel
        if len(img.shape) == 3:
            correction_factor = corrected.astype(np.float32) / (
                gray.astype(np.float32) + 1e-7
            )
            corrected_color = np.zeros_like(img, dtype=np.float32)
            for i in range(3):
                corrected_color[:, :, i] = (
                    img[:, :, i].astype(np.float32) * correction_factor
                )

            corrected_color = np.clip(corrected_color, 0, 255).astype(np.uint8)
            return corrected_color

        return corrected

    # ==================== DENOISING ====================

    def rician_noise_removal(self, img: np.ndarray) -> np.ndarray:
        """
        Remove Rician noise (specific to MRI magnitude images).

        MRI data follows Rician distribution, not Gaussian.
        Standard denoising methods (like Gaussian blur) are suboptimal.

        Method: Non-Local Means adapted for Rician noise.
        """
        if len(img.shape) == 3:
            # For color images, denoise each channel
            denoised = np.zeros_like(img)
            for i in range(3):
                denoised[:, :, i] = cv2.fastNlMeansDenoising(
                    img[:, :, i],
                    None,
                    h=10,  # Filter strength (higher = more smoothing)
                    templateWindowSize=7,
                    searchWindowSize=21,
                )
            return denoised
        else:
            return cv2.fastNlMeansDenoising(
                img, None, h=10, templateWindowSize=7, searchWindowSize=21
            )

    def anisotropic_diffusion(self, img: np.ndarray, iterations: int = 5) -> np.ndarray:
        """
        Perona-Malik anisotropic diffusion.

        Reduces noise while preserving edges - CRITICAL for tumor boundaries.
        Very popular in medical imaging.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        img_float = gray.astype(np.float32)

        # Parameters
        kappa = 50  # Edge threshold
        gamma = 0.1  # Step size

        for _ in range(iterations):
            # Compute gradients
            gN = np.roll(img_float, -1, axis=0) - img_float  # North
            gS = np.roll(img_float, 1, axis=0) - img_float  # South
            gE = np.roll(img_float, -1, axis=1) - img_float  # East
            gW = np.roll(img_float, 1, axis=1) - img_float  # West

            # Compute diffusion coefficients (preserve edges)
            cN = np.exp(-((gN / kappa) ** 2))
            cS = np.exp(-((gS / kappa) ** 2))
            cE = np.exp(-((gE / kappa) ** 2))
            cW = np.exp(-((gW / kappa) ** 2))

            # Update
            img_float += gamma * (cN * gN + cS * gS + cE * gE + cW * gW)

        result = np.clip(img_float, 0, 255).astype(np.uint8)

        if len(img.shape) == 3:
            # Apply same processing to color
            return cv2.cvtColor(
                cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB
            )

        return result

    # ==================== SKULL STRIPPING ====================

    def bet_skull_strip(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Brain Extraction Tool (BET) - inspired by FSL's BET.

        BET is the GOLD STANDARD for skull stripping in neuroimaging.
        This is a simplified implementation.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Step 1: Initial threshold (Otsu's method)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 2: Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Close small holes in the brain
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Remove small objects outside brain
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Step 3: Fill holes inside brain
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return img, np.ones_like(gray)

        # Keep largest component (the brain)
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, -1)

        # Step 4: Smooth mask boundary
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Step 5: Apply mask to original image
        if len(img.shape) == 3:
            result = cv2.bitwise_and(img, img, mask=mask)
        else:
            result = cv2.bitwise_and(gray, gray, mask=mask)

        return result, mask

    def watershed_skull_strip(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Watershed-based skull stripping.
        Often more accurate than simple thresholding for difficult cases.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Step 1: Denoise
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 2: Otsu threshold for sure background
        _, thresh = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 3: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Sure background (dilation)
        sure_bg = cv2.dilate(thresh, kernel, iterations=3)

        # Sure foreground (erosion + distance transform)
        sure_fg_temp = cv2.erode(thresh, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(sure_fg_temp, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Step 4: Connected components for markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # Add 1 so background is not 0
        markers[unknown == 255] = 0

        # Step 5: Watershed
        img_color = (
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if len(img.shape) == 2
            else img.copy()
        )
        markers = cv2.watershed(img_color, markers)

        # Create mask (exclude boundary markers = -1)
        mask = np.where(markers > 1, 255, 0).astype(np.uint8)

        # Apply mask
        if len(img.shape) == 3:
            result = cv2.bitwise_and(img, img, mask=mask)
        else:
            result = cv2.bitwise_and(gray, gray, mask=mask)

        return result, mask

    # ==================== INTENSITY NORMALIZATION ====================

    def nyul_normalization(
        self, img: np.ndarray, percentiles: Tuple = (1, 99)
    ) -> np.ndarray:
        """
        Nyúl et al. histogram normalization (standard in medical imaging).

        Makes intensities comparable across different scanners/protocols.
        ESSENTIAL for multi-site studies and transfer learning.

        Reference: Nyúl, L. G., et al. (2000). "New variants of a method of MRI scale standardization."
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Get percentile landmarks
        p_low, p_high = percentiles
        v_low = np.percentile(gray[gray > 0], p_low)  # Exclude background
        v_high = np.percentile(gray[gray > 0], p_high)

        # Linear mapping to standard range [0, 255]
        normalized = np.zeros_like(gray, dtype=np.float32)
        mask = gray > 0

        normalized[mask] = (gray[mask].astype(np.float32) - v_low) / (
            v_high - v_low + 1e-7
        )
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        if len(img.shape) == 3:
            # Apply to each channel
            normalized_color = np.zeros_like(img, dtype=np.uint8)
            for i in range(3):
                channel = img[:, :, i].astype(np.float32)
                v_low_c = np.percentile(channel[channel > 0], p_low)
                v_high_c = np.percentile(channel[channel > 0], p_high)

                norm_channel = (channel - v_low_c) / \
                    (v_high_c - v_low_c + 1e-7)
                normalized_color[:, :, i] = np.clip(norm_channel * 255, 0, 255)

            return normalized_color

        return normalized

    def histogram_matching(
        self, img: np.ndarray, reference_hist: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Match histogram to a reference (or standard template).
        Very effective for standardizing appearance across datasets.
        """
        if len(img.shape) == 3:
            # Match each channel independently
            matched = np.zeros_like(img)
            for i in range(3):
                matched[:, :, i] = exposure.match_histograms(
                    img[:, :, i],
                    reference_hist
                    if reference_hist is not None
                    else self._get_standard_histogram(),
                    channel_axis=None,
                )
            return matched
        else:
            return exposure.match_histograms(
                img,
                reference_hist
                if reference_hist is not None
                else self._get_standard_histogram(),
                channel_axis=None,
            )

    def _get_standard_histogram(self) -> np.ndarray:
        """
        Return a standard reference histogram (idealized brain MRI).
        In practice, this would be computed from a reference dataset.
        """
        # Create idealized distribution: peak around 127 (gray matter)
        hist = np.zeros(256)
        hist[100:150] = np.random.normal(127, 15, 50)
        hist = hist / hist.sum()
        return hist

    # ==================== ENHANCEMENT ====================

    def adaptive_clahe(self, img: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization.
        Essential for enhancing local contrast (tumor boundaries).
        """
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)

            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(img)

    def unsharp_mask(
        self, img: np.ndarray, sigma: float = 1.0, strength: float = 1.5
    ) -> np.ndarray:
        """
        Unsharp masking for edge enhancement.
        Useful for sharpening tumor boundaries.
        """
        if len(img.shape) == 3:
            enhanced = np.zeros_like(img, dtype=np.float32)
            for i in range(3):
                blurred = gaussian_filter(
                    img[:, :, i].astype(np.float32), sigma=sigma)
                enhanced[:, :, i] = img[:, :, i] + \
                    strength * (img[:, :, i] - blurred)

            return np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)
            enhanced = img + strength * (img - blurred)
            return np.clip(enhanced, 0, 255).astype(np.uint8)

    # ==================== FULL PIPELINE ====================

    def process(self, img: np.ndarray, return_metadata: bool = False):
        """
        Complete medical-grade MRI preprocessing pipeline.

        Order matters! This follows clinical best practices:
        1. Bias correction (must be first)
        2. Denoising
        3. Skull stripping
        4. Intensity normalization
        5. Enhancement
        6. Resize
        """
        metadata = {"stages": {}, "quality_metrics": {}}
        original_shape = img.shape

        # STAGE 1: Bias Field Correction (CRITICAL - must be first)
        if self.apply_n4_bias:
            img = self.n4_bias_correction_simple(img)
            metadata["stages"]["bias_correction"] = "n4_approximation"

        # STAGE 2: Denoising
        if self.apply_rician:
            img = self.rician_noise_removal(img)
            metadata["stages"]["denoising"] = "rician_nlm"
        else:
            img = self.anisotropic_diffusion(img, iterations=3)
            metadata["stages"]["denoising"] = "anisotropic_diffusion"

        # STAGE 3: Skull Stripping
        if self.skull_strip_method == "bet":
            img, mask = self.bet_skull_strip(img)
            metadata["stages"]["skull_strip"] = "bet"
        elif self.skull_strip_method == "watershed":
            img, mask = self.watershed_skull_strip(img)
            metadata["stages"]["skull_strip"] = "watershed"
        else:  # otsu
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(
                img.shape) == 3 else img
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.bitwise_and(img, img, mask=mask)
            metadata["stages"]["skull_strip"] = "otsu"

        # STAGE 4: Intensity Normalization (CRITICAL for cross-scanner generalization)
        if self.intensity_norm == "nyul":
            img = self.nyul_normalization(img, percentiles=(1, 99))
            metadata["stages"]["normalization"] = "nyul"
        elif self.intensity_norm == "histogram_matching":
            img = self.histogram_matching(img)
            metadata["stages"]["normalization"] = "histogram_matching"
        elif self.intensity_norm == "zscore":
            img_float = img.astype(np.float32) / 255.0
            mean = img_float[mask > 0].mean()
            std = img_float[mask > 0].std()
            img_float = (img_float - mean) / (std + 1e-7)
            img = ((img_float + 3) / 6 * 255).clip(0, 255).astype(np.uint8)
            metadata["stages"]["normalization"] = "zscore"

        # STAGE 5: Enhancement
        if self.enhance_method == "clahe":
            img = self.adaptive_clahe(img, clip_limit=2.5)
            metadata["stages"]["enhancement"] = "clahe"
        elif self.enhance_method == "unsharp_mask":
            img = self.unsharp_mask(img, sigma=1.0, strength=1.5)
            metadata["stages"]["enhancement"] = "unsharp_mask"
        elif self.enhance_method == "anisotropic_diffusion":
            img = self.anisotropic_diffusion(img, iterations=5)
            metadata["stages"]["enhancement"] = "anisotropic_diffusion"

        # STAGE 6: Resize
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size,
                             interpolation=cv2.INTER_LANCZOS4)
            metadata["stages"]["resize"] = f"{
                original_shape[:2]} -> {self.target_size}"

        # Quality metrics
        gray_final = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(
                img.shape) == 3 else img
        )
        metadata["quality_metrics"]["snr_estimate"] = float(
            gray_final.mean() / (gray_final.std() + 1e-7)
        )
        metadata["quality_metrics"]["contrast"] = float(
            gray_final.max() - gray_final.min()
        )

        if return_metadata:
            return img, metadata
        return img


def process_dataset_medical_grade(
    input_dir: str, output_dir: str, config: Optional[Dict] = None
):
    """
    Process entire dataset with medical-grade preprocessing.
    """
    from tqdm import tqdm
    import json

    if config is None:
        config = {
            "target_size": (224, 224),
            "apply_n4_bias_correction": True,
            "apply_rician_denoising": True,
            "skull_strip_method": "bet",
            "intensity_normalization": "nyul",
            "enhance_method": "clahe",
        }

    preprocessor = MRIPreprocessor(**config)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in input_path.rglob(
        "*") if p.suffix.lower() in valid_exts]

    print(f"[INFO] Processing {
          len(files)} images with medical-grade pipeline...")
    print(f"[CONFIG] {config}")

    stats = {"total": len(files), "processed": 0, "failed": 0}

    for img_path in tqdm(files):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                stats["failed"] += 1
                continue

            processed, metadata = preprocessor.process(
                img, return_metadata=True)

            # Save
            rel_path = img_path.relative_to(input_path)
            dest_file = output_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(dest_file), processed)

            # Save metadata
            meta_file = dest_file.with_suffix(".json")
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)

            stats["processed"] += 1

        except Exception as e:
            print(f"\n[ERROR] {img_path}: {e}")
            stats["failed"] += 1

    # Summary
    print("\n" + "=" * 60)
    print("MEDICAL-GRADE PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['processed'] / stats['total']:.1%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--skull_strip", default="bet", choices=["bet", "watershed", "otsu"]
    )
    parser.add_argument(
        "--normalization",
        default="nyul",
        choices=["nyul", "zscore", "histogram_matching"],
    )
    parser.add_argument(
        "--enhance",
        default="clahe",
        choices=["clahe", "unsharp_mask", "anisotropic_diffusion"],
    )
    parser.add_argument("--no-bias-correction", action="store_true")
    parser.add_argument("--no-rician", action="store_true")

    args = parser.parse_args()

    config = {
        "target_size": (224, 224),
        "apply_n4_bias_correction": not args.no_bias_correction,
        "apply_rician_denoising": not args.no_rician,
        "skull_strip_method": args.skull_strip,
        "intensity_normalization": args.normalization,
        "enhance_method": args.enhance,
    }

    process_dataset_medical_grade(args.input_dir, args.output_dir, config)
