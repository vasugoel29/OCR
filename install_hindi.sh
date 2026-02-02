#!/bin/bash
# Install Hindi language support for Tesseract OCR

echo "=========================================="
echo "Installing Hindi Language Support"
echo "=========================================="
echo ""

# Check if tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "❌ Tesseract not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr
fi

echo "📦 Installing Hindi language pack..."
sudo apt-get install -y tesseract-ocr-hin tesseract-ocr-script-devanagari

echo ""
echo "✅ Installation complete!"
echo ""
echo "Installed languages:"
tesseract --list-langs

echo ""
echo "=========================================="
echo "Testing Hindi OCR"
echo "=========================================="
echo ""

# Test if Hindi works
if tesseract --list-langs 2>&1 | grep -q "hin"; then
    echo "✅ Hindi language pack installed successfully!"
    echo ""
    echo "You can now process Hindi documents:"
    echo "  python3 ocr.py your_hindi_document.jpg"
    echo "  python3 ocr_json.py your_hindi_document.jpg"
else
    echo "⚠️  Hindi installation may have failed. Please run:"
    echo "  sudo apt-get install tesseract-ocr-hin"
fi

echo ""
echo "=========================================="
