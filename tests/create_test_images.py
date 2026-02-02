"""Create synthetic test images for testing the OCR pipeline."""

import cv2
import numpy as np
from pathlib import Path


def create_test_invoice(output_path: str, quality: str = 'good'):
    """Create a synthetic invoice image for testing.
    
    Args:
        output_path: Path to save the image
        quality: 'good', 'blurry', or 'dark'
    """
    # Create white background
    img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Add invoice content
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Header
    cv2.putText(img, 'ACME CORPORATION', (50, 80), font, 1.5, (0, 0, 0), 2)
    cv2.putText(img, '123 Business St, City, State 12345', (50, 120), font, 0.6, (0, 0, 0), 1)
    
    # Invoice details
    cv2.putText(img, 'INVOICE', (300, 200), font, 2, (0, 0, 0), 3)
    cv2.putText(img, 'Invoice #: INV-2024-001', (50, 280), font, 0.8, (0, 0, 0), 1)
    cv2.putText(img, 'Date: 2024-01-15', (50, 320), font, 0.8, (0, 0, 0), 1)
    cv2.putText(img, 'Due Date: 2024-02-15', (50, 360), font, 0.8, (0, 0, 0), 1)
    
    # Line items
    cv2.line(img, (50, 420), (750, 420), (0, 0, 0), 2)
    cv2.putText(img, 'Description', (50, 450), font, 0.7, (0, 0, 0), 1)
    cv2.putText(img, 'Amount', (600, 450), font, 0.7, (0, 0, 0), 1)
    cv2.line(img, (50, 460), (750, 460), (0, 0, 0), 1)
    
    cv2.putText(img, 'Professional Services', (50, 500), font, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '$1,000.00', (600, 500), font, 0.7, (0, 0, 0), 1)
    
    cv2.putText(img, 'Consulting Fee', (50, 540), font, 0.7, (0, 0, 0), 1)
    cv2.putText(img, '$500.00', (600, 540), font, 0.7, (0, 0, 0), 1)
    
    # Totals
    cv2.line(img, (50, 600), (750, 600), (0, 0, 0), 1)
    cv2.putText(img, 'Subtotal:', (450, 640), font, 0.8, (0, 0, 0), 1)
    cv2.putText(img, '$1,500.00', (600, 640), font, 0.8, (0, 0, 0), 1)
    
    cv2.putText(img, 'Tax (10%):', (450, 680), font, 0.8, (0, 0, 0), 1)
    cv2.putText(img, '$150.00', (600, 680), font, 0.8, (0, 0, 0), 1)
    
    cv2.line(img, (450, 700), (750, 700), (0, 0, 0), 2)
    cv2.putText(img, 'TOTAL:', (450, 740), font, 1.0, (0, 0, 0), 2)
    cv2.putText(img, '$1,650.00', (600, 740), font, 1.0, (0, 0, 0), 2)
    
    # Apply quality degradation
    if quality == 'blurry':
        img = cv2.GaussianBlur(img, (21, 21), 0)
    elif quality == 'dark':
        img = (img * 0.3).astype(np.uint8)
    
    # Save image
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Created {quality} invoice: {output_path}")


def create_test_id(output_path: str, quality: str = 'good'):
    """Create a synthetic ID document for testing.
    
    Args:
        output_path: Path to save the image
        quality: 'good', 'blurry', or 'dark'
    """
    # Create light blue background
    img = np.ones((600, 900, 3), dtype=np.uint8)
    img[:, :] = (240, 220, 200)  # Light blue-gray
    
    # Add border
    cv2.rectangle(img, (20, 20), (880, 580), (0, 0, 0), 3)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Header
    cv2.putText(img, 'GOVERNMENT ID CARD', (250, 80), font, 1.2, (0, 0, 0), 2)
    
    # Photo placeholder (left side)
    cv2.rectangle(img, (50, 120), (250, 380), (100, 100, 100), -1)
    cv2.putText(img, 'PHOTO', (110, 260), font, 1.0, (200, 200, 200), 2)
    
    # ID information (right side)
    y_pos = 150
    line_height = 50
    
    cv2.putText(img, 'Name: John Smith', (300, y_pos), font, 0.8, (0, 0, 0), 1)
    y_pos += line_height
    
    cv2.putText(img, 'ID Number: AB12345678', (300, y_pos), font, 0.8, (0, 0, 0), 1)
    y_pos += line_height
    
    cv2.putText(img, 'Date of Birth: 1990-05-15', (300, y_pos), font, 0.8, (0, 0, 0), 1)
    y_pos += line_height
    
    cv2.putText(img, 'Issue Date: 2020-01-10', (300, y_pos), font, 0.8, (0, 0, 0), 1)
    y_pos += line_height
    
    cv2.putText(img, 'Expiry Date: 2030-01-10', (300, y_pos), font, 0.8, (0, 0, 0), 1)
    
    # Footer
    cv2.putText(img, 'Address: 456 Main Street, Anytown, ST 67890', (50, 480), font, 0.6, (0, 0, 0), 1)
    
    # Apply quality degradation
    if quality == 'blurry':
        img = cv2.GaussianBlur(img, (21, 21), 0)
    elif quality == 'dark':
        img = (img * 0.3).astype(np.uint8)
    
    # Save image
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Created {quality} ID: {output_path}")


def main():
    """Create all test images."""
    output_dir = Path('test_images')
    output_dir.mkdir(exist_ok=True)
    
    print("Creating test images...")
    print("-" * 60)
    
    # Create invoice test images
    create_test_invoice(str(output_dir / 'good_invoice.png'), 'good')
    create_test_invoice(str(output_dir / 'blurry_invoice.png'), 'blurry')
    create_test_invoice(str(output_dir / 'dark_invoice.png'), 'dark')
    
    # Create ID test images
    create_test_id(str(output_dir / 'good_id.png'), 'good')
    create_test_id(str(output_dir / 'blurry_id.png'), 'blurry')
    create_test_id(str(output_dir / 'dark_id.png'), 'dark')
    
    print("-" * 60)
    print(f"✅ Test images created in: {output_dir}/")
    print("\nYou can now test with:")
    print(f"  python test_interactive.py {output_dir}/good_invoice.png")
    print(f"  python -m src.pipeline {output_dir}/good_invoice.png --type invoice")


if __name__ == '__main__':
    main()
