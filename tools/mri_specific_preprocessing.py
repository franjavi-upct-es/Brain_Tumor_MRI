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
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure, filters


class MRIPreprocessor:
    """
    Medical-grade preprocessing from brain MRI images.
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

    # =================== BIAS FIELD CORRECTION ====================

    def n4_bias_correction_simple(self, img: np.ndarray) -> np.ndarray:
        """
        Simplified N4 bias field correction (N4ITK approximation).

        Bias field in MRI causes intensity variations across the image.
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
        sigma = min(img_float.shape) / 8.0  # Adaptative sigma
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

    # =================== SKULL STRIPING ===================

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
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 2: Morphological (Otsu's method)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Close small holes in the brain
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Remove small holes in the brain
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
