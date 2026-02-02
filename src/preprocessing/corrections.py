"""Image preprocessing and correction algorithms.

Implements skew correction, perspective correction, illumination normalization,
and noise removal.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from scipy import ndimage


class ImageCorrector:
    """Handles image preprocessing and corrections."""
    
    def __init__(self, config: dict):
        """Initialize corrector with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.enable_skew = config.get('enable_skew_correction', True)
        self.max_skew_angle = config.get('max_skew_angle', 45)
        self.enable_perspective = config.get('enable_perspective_correction', True)
        self.enable_illumination = config.get('enable_illumination_normalization', True)
        self.enable_noise_removal = config.get('enable_noise_removal', True)
        
        # CLAHE parameters
        self.clahe_clip_limit = config.get('clahe_clip_limit', 2.0)
        self.clahe_tile_grid_size = tuple(config.get('clahe_tile_grid_size', [8, 8]))
        
        # Noise removal parameters
        self.median_blur_ksize = config.get('median_blur_ksize', 3)
        self.bilateral_d = config.get('bilateral_d', 9)
        self.bilateral_sigma_color = config.get('bilateral_sigma_color', 75)
        self.bilateral_sigma_space = config.get('bilateral_sigma_space', 75)
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew using Hough Line Transform.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        if not self.enable_skew:
            return image
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculate angles of all lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)
        
        # Find median angle
        median_angle = np.median(angles)
        
        # Only correct if angle is significant but within max range
        if abs(median_angle) > 0.5 and abs(median_angle) < self.max_skew_angle:
            # Rotate image
            rotated = self._rotate_image(image, median_angle)
            return rotated
        
        return image
    
    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Correct perspective distortion using 4-point transform.
        
        Args:
            image: Input image
            
        Returns:
            Perspective-corrected image
        """
        if not self.enable_perspective:
            return image
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we found a quadrilateral, apply perspective transform
        if len(approx) == 4:
            return self._four_point_transform(image, approx.reshape(4, 2))
        
        return image
    
    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Normalize illumination using CLAHE.
        
        Args:
            image: Input image
            
        Returns:
            Illumination-normalized image
        """
        if not self.enable_illumination:
            return image
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
        
        # Apply to grayscale or each channel
        if len(image.shape) == 2:
            return clahe.apply(image)
        else:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            
            # Convert back to BGR
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using median blur and bilateral filtering.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        if not self.enable_noise_removal:
            return image
        
        # Apply median blur for salt-and-pepper noise
        denoised = cv2.medianBlur(image, self.median_blur_ksize)
        
        # Apply bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(
            denoised,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )
        
        return denoised
    
    def apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization.
        
        Args:
            image: Input image
            
        Returns:
            Binarized image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return binary
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _four_point_transform(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply perspective transform using 4 points.
        
        Args:
            image: Input image
            points: 4 corner points
            
        Returns:
            Transformed image
        """
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = self._order_points(points)
        (tl, tr, br, bl) = rect
        
        # Calculate width of new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Calculate height of new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply transform
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        return warped
    
    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting from top-left.
        
        Args:
            points: 4 corner points
            
        Returns:
            Ordered points array
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference to find corners
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        rect[0] = points[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = points[np.argmax(s)]  # Bottom-right has largest sum
        rect[1] = points[np.argmin(diff)]  # Top-right has smallest difference
        rect[3] = points[np.argmax(diff)]  # Bottom-left has largest difference
        
        return rect
