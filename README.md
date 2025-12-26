# DocRaptor PDF Output Tester

A Flask application for testing different PDF output settings using the DocRaptor API. Perfect for testing print-ready PDF generation with various profiles, ICC color profiles, bleed settings, and card imposition templates.

## Features

- **Multiple PDF Profiles**: PDF/X-4, PDF/X-1a, PDF/X-3, PDF/A variants
- **ICC Color Profile Selection**: GRACoL, SWOP, Fogra, Japan Color, sRGB, or custom URL
- **Bleed Support**: Optional 1/8" bleed with trim marks only (no registration marks)
- **Card Imposition Templates**:
  - **Flat Card**: 2 pages (front with your image, back with simulated content)
  - **Folded Card**: 2 spreads (Panel 4|1 outside, Panel 2|3 inside)
- **Color Options**: True black text, CMYK color conversion
- **Image Fit Modes**: Cover, contain, or fill
- **Test Mode**: Generate watermarked test PDFs without using quota
- **HTML Preview**: View the generated HTML before sending to DocRaptor

## Output Dimensions

- **Trim Size (per panel)**: 4.75" × 6.75"
- **With Bleed**: Image extends 1/8" beyond trim on all sides
- **Trim Marks**: Positioned at 4.75" × 6.75" (trim only, no registration marks)

### Flat Card Layout
- Page 1: Front (your uploaded image)
- Page 2: Back (simulated placeholder content)

### Folded Card Layout
- Spread 1 (Outside): Panel 4 (back cover) | Panel 1 (front cover - your image)
- Spread 2 (Inside): Panel 2 (inside left) | Panel 3 (inside right)
- Each panel: 4.75" × 6.75"
- Full spread: 9.5" × 6.75"

## Installation

```bash
# Clone or download the project
cd docraptor-tester

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
python app.py

# Open in browser
# http://localhost:5000
```

## Configuration Options

### PDF Profiles

| Profile | Description |
|---------|-------------|
| PDF/X-4 | Supports RGB images and transparency. Best for modern print workflows. |
| PDF/X-1a | CMYK only, no transparency. Traditional print compatibility. |
| PDF/X-3 | Allows device-independent color. |
| PDF/A-* | Archive formats for long-term preservation. |

### ICC Profiles

| Profile | Use Case |
|---------|----------|
| GRACoL 2006 Coated | North American coated paper standard |
| SWOP 2006 Coated | Web offset printing standard |
| Fogra39 | European coated paper standard |
| Fogra51 | European coated paper (newer) |
| Japan Color Coated | Japanese printing standard |
| sRGB | Standard RGB for screen display |

### Color Options

- **Use True Black**: Prevents black text from being printed with all 4 CMYK inks (which can cause fuzzy text). Instead, uses only the K (black) channel.
- **Convert CSS Colors to CMYK**: Converts CSS-defined colors to CMYK values in the PDF.

## Recommended Workflow for Print-Ready Cards

Based on the Scodix Design Guide, here's the recommended setup for digital embellishment compatibility:

### For PDF/X-4 with RGB Images (Recommended)

```
Card Type: Flat Card (or Folded Card)
PDF Profile: PDF/X-4
ICC Profile: GRACoL 2006 Coated
Use True Black: ✓ (checked)
Add 1/8" Bleed: ✓ (checked)
Image Fit: Cover
```

This configuration:
1. Preserves RGB images and transparency
2. Embeds a CMYK output intent for print compliance
3. Forces black text to use only the K ink channel
4. Adds proper bleed with trim marks at the correct position (4.75" × 6.75")
5. Generates multi-page PDFs with proper imposition

### Image Preparation

- Keep AI-generated artwork as RGB (sRGB or Display P3)
- Don't pre-convert to CMYK
- Use high-quality JPEG or PNG
- Ensure image extends beyond the trim area for bleed

## API Key

Get your DocRaptor API key from: https://docraptor.com/

The test mode checkbox allows you to generate watermarked test PDFs without using your API quota.

## Project Structure

```
docraptor-tester/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── templates/
    └── index.html     # Web interface
```

## Troubleshooting

### "Invalid API key"
- Make sure you've entered your DocRaptor API key correctly
- Check that your account is active at docraptor.com

### "Image not displaying"
- Ensure the image is a supported format (PNG, JPG, TIFF, WebP)
- Check that the file size is under 50MB

### "PDF generation failed"
- Try enabling "Test Mode" to rule out quota issues
- Check the DocRaptor API status page
- Review the HTML preview for any obvious issues

## License

MIT License - Use freely for your projects.

## Credits

Built for testing Scodix digital embellishment workflows with DocRaptor/Prince XML PDF generation.
