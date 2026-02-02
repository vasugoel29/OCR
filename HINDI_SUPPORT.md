# 🌏 Hindi Language Support

## Installation

### Install Hindi Language Data for Tesseract

```bash
# Install Hindi language pack
sudo apt-get update
sudo apt-get install -y tesseract-ocr-hin tesseract-ocr-script-devanagari

# Verify installation
tesseract --list-langs
```

You should see:
```
List of available languages:
eng
hin
osd
```

## Configuration

The OCR pipeline is already configured to support both English and Hindi!

In [`config.yaml`](file:///home/vasugoel/OCR/config.yaml):
```yaml
ocr:
  language: 'eng+hin'  # Support both English and Hindi
```

### Language Options

- `'eng'` - English only
- `'hin'` - Hindi only  
- `'eng+hin'` - Both English and Hindi (recommended for Indian documents)
- `'hin+eng'` - Hindi priority, then English

## Usage

### No Code Changes Required!

Once Hindi is installed, just use the same scripts:

```bash
# Text output (supports Hindi automatically)
python3 ocr.py your_hindi_document.jpg

# JSON output
python3 ocr_json.py your_hindi_document.jpg
```

### Example Output

For an Aadhaar card with Hindi and English text:

```json
{
  "text": "भारत सरकार Government of India\nआधार Aadhaar\nनाम / Name: राज कुमार / Raj Kumar\nजन्म तिथि / DOB: 15/08/1990",
  "words": 24,
  "confidence": 75.3
}
```

## Testing

### Test with Hindi Documents

```bash
# Test with your Aadhaar card or other Hindi documents
python3 ocr_json.py aadhaar.jpg

# Test with mixed Hindi-English invoice
python3 ocr_json.py invoice_hindi.jpg
```

### Verify Language Support

```bash
# Check installed languages
tesseract --list-langs

# Test Hindi OCR directly
tesseract hindi_image.jpg output -l hin
cat output.txt
```

## Improving Accuracy for Hindi

### 1. Image Quality
- Use high-resolution scans (300+ DPI)
- Ensure good lighting and contrast
- Avoid blurry or skewed images

### 2. Adjust Configuration

Edit `config.yaml` for better Hindi recognition:

```yaml
ocr:
  language: 'hin+eng'  # Prioritize Hindi
  tesseract_config: '--oem 3 --psm 6'  # Try different PSM modes
  min_word_confidence: 30  # Lower threshold for Hindi
```

### 3. PSM Modes for Different Layouts

```yaml
# For single column text
tesseract_config: '--oem 3 --psm 6'

# For sparse text (like ID cards)
tesseract_config: '--oem 3 --psm 11'

# For single line
tesseract_config: '--oem 3 --psm 7'
```

## Common Hindi Documents

### Aadhaar Card
```bash
python3 ocr_json.py aadhaar.jpg
```

Extracts:
- Name (in Hindi and English)
- Aadhaar number
- Date of birth
- Address

### PAN Card
```bash
python3 ocr_json.py pan.jpg
```

### Invoices with Hindi
```bash
python3 ocr_json.py invoice_hindi.jpg
```

## Python API

```python
from src.ocr import TesseractEngine
from src.utils import load_image, load_config

# Load image
image = load_image('hindi_document.jpg')

# Load config (already has Hindi support)
config = load_config('config.yaml')

# Extract text (automatically uses Hindi+English)
ocr = TesseractEngine(config['ocr'])
result = ocr.extract_text(image)

print(result.full_text)  # Contains both Hindi and English text
```

### Custom Language

```python
# Override language for specific use case
config['ocr']['language'] = 'hin'  # Hindi only
ocr = TesseractEngine(config['ocr'])
result = ocr.extract_text(image)
```

## Troubleshooting

### Issue: Hindi text not recognized

**Solution 1: Install Hindi language pack**
```bash
sudo apt-get install tesseract-ocr-hin
```

**Solution 2: Verify language in config**
```yaml
ocr:
  language: 'eng+hin'  # Make sure this is set
```

### Issue: Low accuracy for Hindi

**Solution 1: Adjust PSM mode**
```yaml
tesseract_config: '--oem 3 --psm 6'  # Try different values: 3, 6, 11
```

**Solution 2: Lower confidence threshold**
```yaml
min_word_confidence: 30  # Lower for Hindi (default is 40)
```

**Solution 3: Improve image quality**
- Increase resolution
- Improve lighting
- Remove noise

### Issue: Mixed Hindi-English not working

**Solution: Use combined language**
```yaml
language: 'eng+hin'  # Both languages
```

## Additional Languages

### Install More Languages

```bash
# Marathi
sudo apt-get install tesseract-ocr-mar

# Tamil
sudo apt-get install tesseract-ocr-tam

# Bengali
sudo apt-get install tesseract-ocr-ben

# See all available
apt-cache search tesseract-ocr
```

### Use Multiple Languages

```yaml
ocr:
  language: 'eng+hin+mar'  # English, Hindi, Marathi
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `sudo apt-get install tesseract-ocr-hin` | Install Hindi |
| `tesseract --list-langs` | Check installed languages |
| `python3 ocr.py image.jpg` | Extract text (auto Hindi+English) |
| `python3 ocr_json.py image.jpg` | JSON output with Hindi |

---

**Hindi support is now enabled! Just install the language pack and you're ready to go.** 🎉
