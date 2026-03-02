"""
DocRaptor PDF Output Tester
Flask app for print-ready PDFs (cards, invites, envelopes) via DocRaptor API.
"""

import os
import base64
import tempfile
from io import BytesIO
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import docraptor
from PIL import Image
from collections import Counter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'tiff', 'tif'}

# Directory for storing uploaded ICC profiles
ICC_PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icc_profiles')

# Ensure the directory exists
os.makedirs(ICC_PROFILES_DIR, exist_ok=True)


def get_available_icc_profiles():
    """Get list of available ICC profiles from the icc_profiles directory."""
    profiles = []
    if os.path.exists(ICC_PROFILES_DIR):
        for filename in os.listdir(ICC_PROFILES_DIR):
            if filename.lower().endswith('.icc'):
                profiles.append({
                    'filename': filename,
                    'name': os.path.splitext(filename)[0]
                })
    return sorted(profiles, key=lambda x: x['name'].lower())


def save_icc_profile(file):
    """Save an uploaded ICC profile to the icc_profiles directory."""
    if file and file.filename.lower().endswith('.icc'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(ICC_PROFILES_DIR, filename)
        file.save(filepath)
        return filename
    return None


def get_icc_profile_base64(profile_name):
    """Read an ICC profile from the directory and return as base64."""
    if not profile_name:
        return None
    
    # Add .icc extension if not present
    if not profile_name.lower().endswith('.icc'):
        profile_name = f"{profile_name}.icc"
    
    filepath = os.path.join(ICC_PROFILES_DIR, profile_name)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None

TARGET_DPI = 300

# Target print dimensions (inches) including bleed for each card type
PRINT_DIMENSIONS = {
    'flat': {'panel_w': 5.25, 'panel_h': 7.25},
    'folded': {'panel_w': 5.25, 'panel_h': 7.25, 'spread_w': 10.25, 'spread_h': 7.25},
    'envelope': {'panel_w': 7.5, 'panel_h': 5.5},
}


def ensure_print_dpi(image_data_base64, image_type, target_w_in, target_h_in, target_dpi=TARGET_DPI):
    """Upscale image with LANCZOS if it doesn't meet the target DPI for print."""
    try:
        image_bytes = base64.b64decode(image_data_base64)
        img = Image.open(BytesIO(image_bytes))
        w, h = img.size

        effective_dpi = min(w / target_w_in, h / target_h_in)

        dpi_info = {
            'original_size': f'{w}x{h}',
            'effective_dpi': round(effective_dpi),
            'upscaled': False,
            'warning': None,
        }

        if effective_dpi >= target_dpi:
            dpi_info['status'] = 'ok'
            return image_data_base64, image_type, dpi_info

        scale = max(
            (target_w_in * target_dpi) / w,
            (target_h_in * target_dpi) / h,
        )
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        dpi_info['upscaled'] = True
        dpi_info['new_size'] = f'{new_w}x{new_h}'
        dpi_info['scale_factor'] = round(scale, 2)

        if scale > 2.0:
            dpi_info['status'] = 'low'
            dpi_info['warning'] = (
                f'Very low resolution ({round(effective_dpi)} DPI). '
                f'Upscaled {scale:.1f}x - quality may be poor in print.'
            )
        elif scale > 1.5:
            dpi_info['status'] = 'marginal'
            dpi_info['warning'] = (
                f'Upscaled {scale:.1f}x to reach {target_dpi} DPI. '
                f'Some softening may be visible.'
            )
        else:
            dpi_info['status'] = 'upscaled'

        buf = BytesIO()
        save_fmt = 'PNG' if img.mode == 'RGBA' else 'JPEG'
        save_kwargs = {'quality': 95} if save_fmt == 'JPEG' else {}
        img.save(buf, format=save_fmt, **save_kwargs)
        new_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        new_type = 'png' if save_fmt == 'PNG' else 'jpeg'

        return new_b64, new_type, dpi_info
    except Exception as e:
        print(f"Error in ensure_print_dpi: {e}")
        return image_data_base64, image_type, {
            'effective_dpi': 0, 'upscaled': False, 'warning': str(e), 'status': 'error'
        }


def get_dominant_color(image_data_base64, border_pct=0.12):
    """Extract dominant color by sampling the outer edges of the image.
    
    Analyzes only the outermost border pixels (default 12% on each side)
    to find the background color, ignoring central artwork. Colors are
    quantized to reduce noise from gradients and compression artifacts.
    """
    try:
        image_bytes = base64.b64decode(image_data_base64)
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        w, h = img.size
        
        bx = max(int(w * border_pct), 1)
        by = max(int(h * border_pct), 1)
        
        edge_pixels = []
        for region in [
            img.crop((0, 0, w, by)),           # top strip
            img.crop((0, h - by, w, h)),       # bottom strip
            img.crop((0, by, bx, h - by)),     # left strip
            img.crop((w - bx, by, w, h - by)), # right strip
        ]:
            edge_pixels.extend(region.getdata())
        
        # Quantize to 8-level bins to group similar colors
        quantized = [
            (r >> 5 << 5, g >> 5 << 5, b >> 5 << 5)
            for r, g, b in edge_pixels
        ]
        
        most_common_bin = Counter(quantized).most_common(1)[0][0]
        
        # Average the original pixels that fall into the winning bin
        matching = [
            (r, g, b) for (r, g, b), (qr, qg, qb)
            in zip(edge_pixels, quantized)
            if (qr, qg, qb) == most_common_bin
        ]
        avg_r = sum(r for r, _, _ in matching) // len(matching)
        avg_g = sum(g for _, g, _ in matching) // len(matching)
        avg_b = sum(b for _, _, b in matching) // len(matching)
        
        return f"rgb({avg_r}, {avg_g}, {avg_b})"
    except Exception as e:
        print(f"Error extracting dominant color: {e}")
        return "rgb(245, 245, 240)"

# PDF profiles available in DocRaptor/Prince
PDF_PROFILES = [
    'PDF/X-4',
    'PDF/X-1a',
    'PDF/X-3',
    'PDF/A-1a',
    'PDF/A-1b',
    'PDF/A-3a',
    'PDF/A-3b',
    'PDF/UA-1',
    ''  # Default (no profile)
]

def get_branding_svg(heart_color='#bd2231', text_color='#ffffff'):
    """Generate HeartStamp branding SVG with configurable colors."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" fill="{heart_color}" viewBox="0 0 140 35"><g clip-path="url(#hs_clip)"><path d="M32.28 3.2Q29.458 0 25.295 0q-4.199 0-7.655 4.04l-.16.173-.159-.173Q13.865 0 9.666 0 5.505 0 2.68 3.2.001 6.298 0 10.518q0 6.262 3.704 12.558 3.562 6.12 8.961 9.674.741.49 2.258 1.301c1.223.633 2.275.95 2.557.95s1.335-.317 2.558-.95q1.517-.81 2.258-1.3 5.4-3.555 8.961-9.675 3.704-6.295 3.704-12.558 0-4.22-2.681-7.316"/><path fill="{text_color}" d="M38.3 19.36c0 .418-.039.798-.066 1.076a.143.143 0 0 1-.143.128h-8.634a.144.144 0 0 0-.143.162c.325 2.473 1.765 3.529 3.376 3.529 1.089 0 1.964-.38 2.83-1.001a.145.145 0 0 1 .19.018l1.385 1.493a.144.144 0 0 1-.007.203c-1.162 1.067-2.554 1.67-4.552 1.67-3.165 0-5.84-2.537-5.84-6.996 0-4.562 2.418-7.022 5.917-7.022 3.834 0 5.686 3.101 5.686 6.74m-2.833-.974a.144.144 0 0 0 .144-.155c-.167-1.776-1.053-3.228-3.178-3.228-1.73 0-2.8 1.169-3.068 3.222a.144.144 0 0 0 .144.161zM50.325 26.356h-2.311a.144.144 0 0 1-.144-.143v-.857a.143.143 0 0 0-.24-.106c-.907.807-2.12 1.388-3.465 1.388-2.187 0-4.683-1.23-4.683-4.536 0-2.998 2.316-4.356 5.378-4.356 1.154 0 2.093.15 2.816.431a.143.143 0 0 0 .194-.134v-.784c0-1.461-.9-2.281-2.547-2.281-1.333 0-2.382.236-3.396.775a.144.144 0 0 1-.196-.06l-.9-1.749a.144.144 0 0 1 .053-.19c1.257-.748 2.669-1.134 4.516-1.134 3.01 0 5.069 1.46 5.069 4.51v9.083c0 .08-.065.143-.144.143M47.87 22.49v-1.783a.14.14 0 0 0-.081-.13c-.71-.336-1.62-.55-3.007-.55-1.698 0-2.778.77-2.778 2 0 1.332.849 2.229 2.598 2.229 1.394 0 2.613-.845 3.24-1.679a.14.14 0 0 0 .028-.087M60.988 13.286l-.62 2.239a.143.143 0 0 1-.2.09c-.521-.258-1.083-.407-1.879-.407-1.672 0-2.65 1.18-2.65 3.46v7.545c0 .08-.064.143-.144.143h-2.362a.144.144 0 0 1-.144-.143V13.045c0-.079.065-.143.144-.143h2.362c.08 0 .144.064.144.143v.724c0 .132.163.193.25.095.65-.732 1.664-1.244 2.838-1.244 1.021 0 1.675.183 2.2.507a.14.14 0 0 1 .061.16M69.445 23.965l-.263 1.985a.14.14 0 0 1-.074.108c-.639.34-1.514.58-2.535.58-1.878 0-3.035-1.153-3.035-3.562v-7.622a.144.144 0 0 0-.144-.143H61.66a.144.144 0 0 1-.14-.178l.521-2.122a.144.144 0 0 1 .14-.11h1.213c.079 0 .144-.063.144-.143V9.144c0-.053.029-.102.076-.127a5.8 5.8 0 0 0 2.31-1.23.15.15 0 0 1 .142.003c.04.024.07.069.07.123v4.845c0 .08.064.144.144.144h3.057c.08 0 .144.064.144.143v2.122c0 .08-.064.144-.144.144H66.28a.144.144 0 0 0-.144.143v7.16c0 1.256.412 1.615 1.39 1.615.59 0 1.235-.185 1.716-.413a.143.143 0 0 1 .203.149M84.249 21.205c0 3.23-2.059 5.433-6.664 5.433-2.573 0-4.8-1.084-6.317-2.793a.144.144 0 0 1 .009-.198l1.727-1.674a.144.144 0 0 1 .204.004C74.414 23.223 76.138 24 77.79 24c2.521 0 3.73-.871 3.73-2.614 0-1.384-1.055-2.076-4.065-2.973-3.808-1.127-5.634-2.075-5.634-5.279 0-3.1 2.624-4.997 5.943-4.997 2.39 0 4.214.857 5.785 2.333.059.055.06.148.003.206l-1.699 1.715a.144.144 0 0 1-.203 0c-1.108-1.084-2.37-1.615-4.092-1.615-2.11 0-3.01 1.025-3.01 2.23 0 1.256.823 1.87 3.936 2.793 3.55 1.077 5.764 2.204 5.764 5.408M92.96 23.965l-.263 1.985a.14.14 0 0 1-.074.108c-.639.34-1.514.58-2.535.58-1.878 0-3.036-1.153-3.036-3.562v-7.622a.144.144 0 0 0-.144-.143h-1.641a.144.144 0 0 1-.144-.144v-2.122c0-.079.064-.143.144-.143h1.641c.08 0 .144-.064.144-.144V9.144c0-.053.03-.102.076-.127a4.6 4.6 0 0 0 2.311-1.23.15.15 0 0 1 .142.003c.04.024.07.069.07.123v4.845c0 .08.064.144.144.144h3.057c.08 0 .144.064.144.143v2.122c0 .08-.065.144-.144.144h-3.057a.144.144 0 0 0-.144.143v7.16c0 1.256.412 1.615 1.39 1.615.59 0 1.235-.185 1.716-.413a.143.143 0 0 1 .203.149M105.124 26.356h-2.311a.144.144 0 0 1-.144-.143v-.857a.144.144 0 0 0-.24-.106c-.906.807-2.121 1.388-3.465 1.388-2.186 0-4.682-1.23-4.682-4.536 0-2.998 2.315-4.356 5.377-4.356 1.154 0 2.093.15 2.816.431a.143.143 0 0 0 .194-.134v-.784c0-1.461-.9-2.281-2.547-2.281-1.332 0-2.381.236-3.395.775a.144.144 0 0 1-.197-.06l-.9-1.749a.144.144 0 0 1 .054-.19c1.257-.748 2.668-1.134 4.516-1.134 3.01 0 5.068 1.46 5.068 4.51v9.083c0 .08-.065.143-.144.143m-2.455-3.867v-1.783a.14.14 0 0 0-.081-.13c-.709-.336-1.619-.55-3.006-.55-1.698 0-2.779.77-2.779 2 0 1.332.85 2.229 2.599 2.229 1.393 0 2.613-.845 3.239-1.679a.14.14 0 0 0 .028-.087M125.989 26.356h-2.362a.143.143 0 0 1-.144-.143v-7.468c0-2.717-.849-3.69-2.599-3.69-1.775 0-2.598 1.255-2.598 3.434v7.724c0 .079-.065.143-.144.143h-2.362a.143.143 0 0 1-.144-.143v-7.468c0-2.717-.849-3.69-2.599-3.69-1.775 0-2.599 1.255-2.599 3.434v7.724c0 .079-.064.143-.144.143h-2.362a.144.144 0 0 1-.144-.143V13.045c0-.079.065-.143.144-.143h2.362c.08 0 .144.064.144.143v.722c0 .132.164.194.251.095.688-.78 1.736-1.242 2.966-1.242 1.743 0 2.877.63 3.603 1.802a.144.144 0 0 0 .232.014c.943-1.114 1.996-1.816 4.012-1.816 3.138 0 4.631 2.05 4.631 6.022v7.57c0 .08-.065.144-.144.144M140 19.847c0 4.613-2.521 6.791-5.326 6.791-1.227 0-2.391-.58-3.156-1.294-.091-.085-.24-.018-.24.106v4.407c0 .054-.03.103-.077.128a4.44 4.44 0 0 0-2.337 1.209c-.004.002-.073.036-.141-.005a.14.14 0 0 1-.069-.123v-18.02c0-.08.064-.144.144-.144h2.336c.079 0 .144.064.144.144v.819c0 .122.144.188.237.108.923-.789 1.986-1.353 3.288-1.353 2.907 0 5.197 2.101 5.197 7.227m-2.65.077c0-3.383-1.081-4.87-3.036-4.87-1.242 0-2.333.821-3.03 1.678a.14.14 0 0 0-.032.091v5.564q0 .048.029.087c.603.787 1.82 1.73 3.136 1.73 1.904 0 2.933-1.435 2.933-4.28M24.178 8.415h-2.413a.144.144 0 0 0-.144.144v6.962a.143.143 0 0 1-.144.144 10.284 10.284 0 0 1-7.947.02l.003-.004q-.083-.038-.164-.078a.14.14 0 0 1-.029-.082v-7.12h-.002c-.058-2.91-2.844-3.24-4.52-2.81-2.06.528-3.776 3.207-2.677 6.038a10.02 10.02 0 0 0 4.499 5.412v9.162c0 .08.064.144.144.144h2.412c.08 0 .144-.064.144-.144v-7.705a.143.143 0 0 1 .144-.144 10.1 10.1 0 0 0 3.819.826 10.1 10.1 0 0 0 4.175-.826c.08 0 .143.065.143.144v7.705c0 .08.065.144.144.144h2.413c.08 0 .144-.065.144-.144V8.56a.144.144 0 0 0-.144-.144m-13.944 4.878A10.2 10.2 0 0 1 8.142 9.84c-.258-.802.177-1.14.51-1.166.398-.031.512.248.746 1.023.003.011.02.007.018-.004-.07-.42-.104-1.16.31-1.418.257-.159.57-.16.753.05.035.039.16.163.16.446v4.938q-.232-.226-.405-.416"/></g><defs><clipPath id="hs_clip"><path fill="#fff" d="M0 0h140v35H0z"/></clipPath></defs></svg>'''


# Simulated inside panel content for folded cards
INSIDE_LEFT_CONTENT = """
<div class="inside-content inside-left">
    <p class="small-text">Panel 2 - Inside Left</p>
</div>
"""

INSIDE_RIGHT_CONTENT = """
<div class="inside-content inside-right">
    <div class="message-area">
        <p class="preprinted-message">Wishing you all the best on your special day!</p>
        <div class="signature-line">
            <span>With love,</span>
            <div class="line"></div>
        </div>
    </div>
    <p class="small-text">Panel 3 - Inside Right</p>
</div>
"""

PANEL_4_CONTENT = """
<div class="back-content panel-4">
    <div class="logo-placeholder">✦</div>
    <h2>Premium Greeting Card</h2>
    <p class="tagline">Crafted with care</p>
    <div class="details">
        <p>Made in USA • Recycled Paper</p>
        <p>www.example.com</p>
    </div>
    <div class="barcode-placeholder">
        <div class="barcode-lines"></div>
        <span>1234567890</span>
    </div>
    <p class="small-text">Panel 4 - Back Cover</p>
</div>
"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_prince_pdf_css(settings):
    """Generate @prince-pdf CSS block based on settings."""
    prince_pdf_css = []
    
    # PDF Profile - set explicitly in CSS as well as prince_options
    pdf_profile = settings.get('pdf_profile', 'PDF/X-4')
    if pdf_profile:
        prince_pdf_css.append(f'prince-pdf-profile: "{pdf_profile}";')
    
    # Color options
    color_options = []
    if settings.get('use_true_black', True):
        color_options.append('use-true-black')
    if settings.get('use_cmyk_colors', False):
        color_options.append('use-cmyk-colors')
    if color_options:
        prince_pdf_css.append(f"prince-pdf-color-options: {' '.join(color_options)};")
    
    # Output intent (ICC profile) - required for PDF/X compliance
    # Use base64-embedded ICC profile from uploaded files
    icc_base64 = settings.get('icc_base64')
    if icc_base64:
        # Use embedded base64 data URI - most reliable method
        prince_pdf_css.append(f'prince-pdf-output-intent: url("data:application/vnd.iccprofile;base64,{icc_base64}");')
    
    if prince_pdf_css:
        return f"""
        @prince-pdf {{
            {chr(10).join('            ' + css for css in prince_pdf_css)}
        }}
        """
    return ''


def get_common_styles():
    """Return common CSS styles for all templates."""
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .panel-content {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        
        .image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center;
        }
        
        /* Inside panel styles */
        .inside-content {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 0.5in;
            font-family: 'Georgia', serif;
            color: #333;
        }
        
        .inside-content .message-area {
            max-width: 80%;
        }
        
        .inside-content .preprinted-message {
            font-size: 14pt;
            font-style: italic;
            line-height: 1.8;
            margin-bottom: 0.5in;
        }
        
        .inside-content .signature-line {
            text-align: left;
        }
        
        .inside-content .signature-line span {
            font-size: 12pt;
            display: block;
            margin-bottom: 0.1in;
        }
        
        .inside-content .signature-line .line {
            width: 2in;
            border-bottom: 1px solid #ccc;
            height: 0.3in;
        }
        
        .small-text {
            position: absolute;
            bottom: 0.15in;
            font-size: 6pt;
            color: #ccc;
        }
    """


def generate_flat_card_html(image_data, image_type, settings, back_image_data=None, back_image_type=None):
    """Generate HTML for flat card (2 pages: front and back)."""
    
    # Card dimensions (trim size)
    card_width = 5.0   # inches
    card_height = 7.0  # inches
    bleed = 0.125 if settings.get('add_bleed', False) else 0  # 1/8 inch
    
    # Total dimensions with bleed
    total_width = card_width + (bleed * 2)
    total_height = card_height + (bleed * 2)
    
    prince_pdf_block = get_prince_pdf_css(settings)
    fit_mode = settings.get('image_fit', 'cover')
    bg_color = settings.get('background_color', '#ffffff')
    
    marks = 'crop' if settings.get('include_crop_marks', False) and bleed > 0 else 'none'
    
    # Determine back panel content
    if back_image_data and back_image_type:
        back_bg_content = f'<img class="image" src="data:image/{back_image_type};base64,{back_image_data}" alt="Card Back">'
        back_bg_color = 'transparent'
    else:
        back_bg_content = ''
        back_bg_color = bg_color
    
    # Branding overlay for flat card back panel
    include_branding = settings.get('include_branding', True)
    branding_height = float(settings.get('branding_height', 1.0))
    heart_color = settings.get('heart_color', '#bd2231')
    text_color = settings.get('text_color', '#ffffff')
    branding_svg = get_branding_svg(heart_color=heart_color, text_color=text_color) if include_branding else None
    branding_bg_html = ''
    branding_img_html = ''
    branding_css = ''
    
    if branding_svg:
        branding_bg_html = '<div class="branding-bg"></div>'
        branding_img_html = f'<div class="branding-img">{branding_svg}</div>'
        logo_size = float(settings.get('branding_logo_size', 0.30))
        logo_size = max(0.05, min(1.0, logo_size))
        branding_logo_width = card_width * logo_size
        branding_logo_left = (card_width - branding_logo_width) / 2
        bg_pad_h = 0.04
        bg_pad_v = 0.02
        branding_bg_width = branding_logo_width + (bg_pad_h * 2)
        branding_bg_left = (card_width - branding_bg_width) / 2
        branding_bg_height = branding_height + bg_pad_v + bleed
        branding_css = f"""
        /* Branding colored background - centered strip with padding */
        .branding-bg {{
            position: absolute;
            bottom: -{bleed}in;
            left: {branding_bg_left}in;
            width: {branding_bg_width}in;
            height: {branding_bg_height}in;
            background-color: {bg_color};
            border-radius: 0.04in 0.04in 0 0;
            z-index: 2;
        }}
        
        /* Branding logo overlay - narrow, centered at bottom */
        .branding-img {{
            position: absolute;
            bottom: 0;
            left: {branding_logo_left}in;
            width: {branding_logo_width}in;
            height: {branding_height}in;
            z-index: 3;
        }}
        
        .branding-img svg {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        {prince_pdf_block}
        
        @page {{
            size: {card_width}in {card_height}in;
            margin: 0;
            bleed: {bleed}in;
            marks: {marks};
            prince-pdf-page-colorspace: rgb;
        }}
        
        {get_common_styles()}
        
        html, body {{
            margin: 0;
            padding: 0;
        }}
        
        .page {{
            position: relative;
            width: {card_width}in;
            height: {card_height}in;
            page-break-after: always;
            overflow: visible;
        }}
        
        .page:last-child {{
            page-break-after: avoid;
        }}
        
        /* Content box extends into bleed area using negative positioning */
        .page-content {{
            position: absolute;
            top: -{bleed}in;
            left: -{bleed}in;
            width: {total_width}in;
            height: {total_height}in;
            background-color: {bg_color};
        }}
        
        .image {{
            width: 100%;
            height: 100%;
            object-fit: {fit_mode};
            object-position: center;
            display: block;
        }}
        
        /* Back page: background extends into bleed */
        .back-page-bg {{
            position: absolute;
            top: -{bleed}in;
            left: -{bleed}in;
            width: {total_width}in;
            height: {total_height}in;
            background-color: {back_bg_color};
            z-index: 1;
        }}
        
        {branding_css}
    </style>
</head>
<body>
    <!-- Page 1: Front (uploaded image) -->
    <div class="page">
        <div class="page-content">
            <img class="image" src="data:image/{image_type};base64,{image_data}" alt="Card Front">
        </div>
    </div>
    
    <!-- Page 2: Back -->
    <div class="page">
        <div class="back-page-bg">
            {back_bg_content}
        </div>
        {branding_bg_html}
        {branding_img_html}
    </div>
</body>
</html>"""
    
    return html


def generate_folded_card_html(image_data, image_type, settings, inside_image_data=None, inside_image_type=None, back_image_data=None, back_image_type=None):
    """Generate HTML for folded card (2 spreads: outside and inside)."""
    
    # Single panel dimensions (trim size)
    panel_width = 5.0   # inches
    panel_height = 7.0  # inches
    bleed = 0.125 if settings.get('add_bleed', False) else 0  # 1/8 inch
    
    # Spread dimensions (2 panels side by side) - this is the TRIM size
    spread_width = panel_width * 2  # 10 inches
    spread_height = panel_height     # 7 inches
    
    # Total dimensions with bleed (content extends beyond trim)
    total_spread_width = spread_width + (bleed * 2)   # 10.25 inches with bleed
    total_spread_height = spread_height + (bleed * 2)  # 7.25 inches with bleed
    
    prince_pdf_block = get_prince_pdf_css(settings)
    fit_mode = settings.get('image_fit', 'cover')
    bg_color = settings.get('background_color', '#ffffff')
    
    marks = 'crop' if settings.get('include_crop_marks', False) and bleed > 0 else 'none'
    
    # Determine back panel content (Panel 4)
    if back_image_data and back_image_type:
        back_panel_content = f'<img class="image" src="data:image/{back_image_type};base64,{back_image_data}" alt="Back Cover">'
    else:
        # Extract dominant color from front panel image for Panel 4 background
        dominant_color = get_dominant_color(image_data)
        back_panel_content = f'<div style="width: 100%; height: 100%; background-color: {dominant_color};"></div>'
    
    # Determine inside panel content (spans both Panel 2 and Panel 3)
    if inside_image_data and inside_image_type:
        # Single image spans both inside panels - left panel shows left half, right panel shows right half
        inside_left_content = f'<img class="image" src="data:image/{inside_image_type};base64,{inside_image_data}" alt="Inside Left" style="object-position: left center;">'
        inside_right_content = f'<img class="image" src="data:image/{inside_image_type};base64,{inside_image_data}" alt="Inside Right" style="object-position: right center;">'
        # No fold indicator needed when using custom inside images
        inside_fold_indicator = ''
    else:
        inside_left_content = INSIDE_LEFT_CONTENT
        inside_right_content = INSIDE_RIGHT_CONTENT
        # Show fold indicator for placeholder content
        inside_fold_indicator = '<div class="fold-indicator"></div>'
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        {prince_pdf_block}
        
        @page {{
            size: {spread_width}in {spread_height}in;
            margin: 0;
            bleed: {bleed}in;
            marks: {marks};
            prince-pdf-page-colorspace: rgb;
        }}
        
        {get_common_styles()}
        
        html, body {{
            margin: 0;
            padding: 0;
        }}
        
        .spread {{
            position: relative;
            width: {spread_width}in;
            height: {spread_height}in;
            page-break-after: always;
            overflow: visible;
        }}
        
        .spread:last-child {{
            page-break-after: avoid;
        }}
        
        /* Content extends into bleed area */
        .spread-content {{
            position: absolute;
            top: -{bleed}in;
            left: -{bleed}in;
            width: {total_spread_width}in;
            height: {total_spread_height}in;
            display: flex;
            flex-direction: row;
            background-color: {bg_color};
        }}
        
        /* Each panel takes half the spread (plus outer bleed) */
        .panel {{
            width: 50%;
            height: 100%;
            position: relative;
            overflow: hidden;
        }}
        
        .panel-inner {{
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            background-color: {bg_color};
        }}
        
        .image {{
            width: 100%;
            height: 100%;
            object-fit: {fit_mode};
            object-position: center;
            display: block;
        }}
        
        /* Fold line indicator (visual guide on trim, very subtle) */
        .fold-indicator {{
            position: absolute;
            top: 0;
            left: 50%;
            width: 0;
            height: 100%;
            border-left: 0.5px dashed rgba(200, 200, 200, 0.5);
            z-index: 10;
        }}
    </style>
</head>
<body>
    <!-- Spread 1: Outside (Panel 4 left | Panel 1 right) -->
    <!-- When printed and folded, Panel 1 becomes front cover, Panel 4 becomes back -->
    <div class="spread">
        <div class="spread-content">
            <!-- Panel 4: Back Cover (left side of spread) -->
            <div class="panel">
                <div class="panel-inner">
                    {back_panel_content}
                </div>
            </div>
            <!-- Panel 1: Front Cover (right side of spread) - uploaded image -->
            <div class="panel">
                <div class="panel-inner">
                    <img class="image" src="data:image/{image_type};base64,{image_data}" alt="Front Cover">
                </div>
            </div>
        </div>
        <div class="fold-indicator"></div>
    </div>
    
    <!-- Spread 2: Inside (Panel 2 left | Panel 3 right) -->
    <div class="spread">
        <div class="spread-content">
            <!-- Panel 2: Inside Left -->
            <div class="panel">
                <div class="panel-inner">
                    {inside_left_content}
                </div>
            </div>
            <!-- Panel 3: Inside Right -->
            <div class="panel">
                <div class="panel-inner">
                    {inside_right_content}
                </div>
            </div>
        </div>
        {inside_fold_indicator}
    </div>
</body>
</html>"""
    
    return html


def generate_envelope_html(image_data, image_type, settings):
    """Generate HTML for A7 envelope front panel with USPS zone compliance.
    
    image_data and image_type can be None if no background image is provided;
    the envelope will have a transparent background (ink-only output).
    """
    
    # A7 envelope dimensions (trim size, landscape)
    env_width = 7.25   # inches
    env_height = 5.25  # inches
    bleed = 0.125 if settings.get('add_bleed', False) else 0
    
    total_width = env_width + (bleed * 2)
    total_height = env_height + (bleed * 2)
    
    prince_pdf_block = get_prince_pdf_css(settings)
    fit_mode = settings.get('image_fit', 'cover')
    marks = 'crop' if settings.get('include_crop_marks', False) and bleed > 0 else 'none'
    
    # Envelope-specific settings
    envelope = settings.get('envelope', {})
    return_name = envelope.get('return_name', 'JOHN DOE')
    return_address = envelope.get('return_address', '123 MAIN STREET\nANYTOWN, ST 12345')
    delivery_name = envelope.get('delivery_name', 'JANE SMITH')
    delivery_address = envelope.get('delivery_address', '456 OAK AVENUE APT 2B\nSOMEWHERE, ST 67890')
    text_color = envelope.get('text_color', '#000000')
    font_family = envelope.get('font_family', 'Caveat')
    
    # Build return address lines
    return_lines = f'<div class="address-name">{return_name}</div>'
    for line in return_address.split('\n'):
        line = line.strip()
        if line:
            return_lines += f'<div>{line}</div>'
    
    # Build delivery address lines
    delivery_lines = f'<div class="address-name">{delivery_name}</div>'
    for line in delivery_address.split('\n'):
        line = line.strip()
        if line:
            delivery_lines += f'<div>{line}</div>'
    
    # Google Fonts import for the selected font
    font_import = f"@import url('https://fonts.googleapis.com/css2?family={font_family.replace(' ', '+')}:wght@400;700&display=swap');"
    
    # Background: image if provided, otherwise transparent
    if image_data and image_type:
        bg_html = f'''<div class="envelope-bg">
            <img src="data:image/{image_type};base64,{image_data}" alt="Envelope Background">
        </div>'''
        bg_css = f""".envelope-bg {{
            position: absolute;
            top: -{bleed}in;
            left: -{bleed}in;
            width: {total_width}in;
            height: {total_height}in;
        }}
        
        .envelope-bg img {{
            width: 100%;
            height: 100%;
            object-fit: {fit_mode};
            object-position: center;
            display: block;
        }}"""
    else:
        bg_html = ''
        bg_css = ''
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        {font_import}
        
        {prince_pdf_block}
        
        @page {{
            size: {env_width}in {env_height}in;
            margin: 0;
            bleed: {bleed}in;
            marks: {marks};
            prince-pdf-page-colorspace: rgb;
        }}
        
        {get_common_styles()}
        
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
        }}
        
        .envelope {{
            position: relative;
            width: {env_width}in;
            height: {env_height}in;
            overflow: visible;
            background: transparent;
        }}
        
        {bg_css}
        
        .envelope-content {{
            position: absolute;
            top: 0;
            left: 0;
            width: {env_width}in;
            height: {env_height}in;
            font-family: '{font_family}', cursive, sans-serif;
            color: {text_color};
        }}
        
        /* Zone 2: Return address - upper left, within top 2.5", max 3.625" wide */
        .zone-return {{
            position: absolute;
            top: 0.375in;
            left: 0.5in;
            max-width: 3.125in;
            font-size: 11pt;
            line-height: 1.5;
        }}
        
        .zone-return .address-name {{
            font-weight: 700;
            margin-bottom: 2pt;
        }}
        
        /* Zone 3: OCR Read Area (Delivery) */
        .zone-delivery {{
            position: absolute;
            left: 3.0in;
            bottom: 1.25in;
            width: 3.75in;
            font-size: 11pt;
            line-height: 1.6;
        }}
        
        .zone-delivery .address-name {{
            font-weight: 700;
            margin-bottom: 2pt;
        }}
        
        /* Zone 4: Barcode clear - bottom 0.625", right 4.75" - NO INK */
    </style>
</head>
<body>
    <div class="envelope">
        {bg_html}
        
        <div class="envelope-content">
            <div class="zone-return">
                {return_lines}
            </div>
            
            <div class="zone-delivery">
                {delivery_lines}
            </div>
        </div>
    </div>
</body>
</html>"""
    
    return html


def generate_html_for_image(image_data, image_type, settings, additional_images=None):
    """Generate HTML document for DocRaptor based on card type.
    
    Args:
        image_data: Base64 encoded front image data
        image_type: MIME type of front image
        settings: Dictionary of settings
        additional_images: Dictionary with optional 'back' and 'inside' image data
            - back: {'data': base64_data, 'type': image_type}
            - inside: {'data': base64_data, 'type': image_type}
    """
    additional_images = additional_images or {}
    card_type = settings.get('card_type', 'flat')
    
    # Extract additional image data
    back_data = additional_images.get('back', {}).get('data')
    back_type = additional_images.get('back', {}).get('type')
    inside_data = additional_images.get('inside', {}).get('data')
    inside_type = additional_images.get('inside', {}).get('type')
    
    if card_type == 'folded':
        return generate_folded_card_html(
            image_data, image_type, settings,
            inside_image_data=inside_data,
            inside_image_type=inside_type,
            back_image_data=back_data,
            back_image_type=back_type
        )
    elif card_type == 'envelope':
        return generate_envelope_html(image_data, image_type, settings)
    else:
        return generate_flat_card_html(
            image_data, image_type, settings,
            back_image_data=back_data,
            back_image_type=back_type
        )


def create_pdf(html_content, settings, api_key):
    """Create PDF using DocRaptor API."""
    
    # Initialize DocRaptor client
    doc_api = docraptor.DocApi()
    doc_api.api_client.configuration.username = api_key
    
    # Build prince_options
    prince_options = {
        'css_dpi': 300  # Set DPI for print quality
    }
    
    # PDF profile (PDF/X-4, PDF/X-1a, etc.)
    pdf_profile = settings.get('pdf_profile', 'PDF/X-4')
    if pdf_profile:
        prince_options['profile'] = pdf_profile
    
    # Output Intent is handled via CSS @prince-pdf with base64-embedded ICC profile
    # No URL fallback needed since we only use uploaded profiles
    
    # PDF version
    pdf_version = settings.get('pdf_version')
    if pdf_version:
        prince_options['pdf_version'] = pdf_version
    
    # Color conversion
    if settings.get('force_cmyk', False):
        prince_options['force_identity_encoding'] = False
    
    # Build the document request
    doc_params = {
        'name': 'docraptor-test.pdf',
        'document_type': 'pdf',
        'document_content': html_content,
        'test': settings.get('test_mode', True),  # Set to False for production
    }
    
    if prince_options:
        doc_params['prince_options'] = prince_options
    
    try:
        # Create the document
        response = doc_api.create_doc(doc_params)
        return response, None
    except docraptor.rest.ApiException as e:
        return None, str(e)


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', 
                         pdf_profiles=PDF_PROFILES,
                         icc_profiles=get_available_icc_profiles())


@app.route('/api/icc-profiles', methods=['GET'])
def list_icc_profiles():
    """API endpoint to list available ICC profiles."""
    return jsonify({'profiles': get_available_icc_profiles()})


@app.route('/api/icc-profiles', methods=['POST'])
def upload_icc_profile():
    """API endpoint to upload a new ICC profile."""
    if 'icc_file' not in request.files:
        return jsonify({'error': 'No ICC file provided'}), 400
    
    file = request.files['icc_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.icc'):
        return jsonify({'error': 'File must have .icc extension'}), 400
    
    filename = save_icc_profile(file)
    if filename:
        return jsonify({
            'success': True,
            'filename': filename,
            'profiles': get_available_icc_profiles()
        })
    else:
        return jsonify({'error': 'Failed to save ICC profile'}), 500


@app.route('/api/icc-profiles/<filename>', methods=['DELETE'])
def delete_icc_profile(filename):
    """API endpoint to delete an ICC profile."""
    filepath = os.path.join(ICC_PROFILES_DIR, secure_filename(filename))
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({
            'success': True,
            'profiles': get_available_icc_profiles()
        })
    else:
        return jsonify({'error': 'Profile not found'}), 404


@app.route('/generate', methods=['POST'])
def generate_pdf():
    """Generate PDF with the specified settings."""
    
    # Get API key
    api_key = request.form.get('api_key', '').strip()
    if not api_key:
        return jsonify({'error': 'DocRaptor API key is required'}), 400
    
    card_type = request.form.get('card_type', 'flat')
    
    # Get uploaded image (optional for envelope)
    image_data = None
    image_type = None
    front_image_name = 'envelope'
    
    has_image = 'image' in request.files and request.files['image'].filename != ''
    
    if has_image:
        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        front_image_name = os.path.splitext(secure_filename(file.filename))[0]
        image_data = base64.b64encode(file.read()).decode('utf-8')
        image_type = file.filename.rsplit('.', 1)[1].lower()
        if image_type == 'jpg':
            image_type = 'jpeg'
    elif card_type != 'envelope':
        return jsonify({'error': 'No image file provided'}), 400
    
    # Process additional images if provide_all_images is checked
    additional_images = {}
    provide_all = request.form.get('provide_all_images') == 'true'
    
    if provide_all:
        # Get back image
        if 'back_image' in request.files:
            back_file = request.files['back_image']
            if back_file.filename != '' and allowed_file(back_file.filename):
                back_data = base64.b64encode(back_file.read()).decode('utf-8')
                back_type = back_file.filename.rsplit('.', 1)[1].lower()
                if back_type == 'jpg':
                    back_type = 'jpeg'
                additional_images['back'] = {'data': back_data, 'type': back_type}
        
        # Get inside image (only for folded cards)
        if card_type == 'folded' and 'inside_image' in request.files:
            inside_file = request.files['inside_image']
            if inside_file.filename != '' and allowed_file(inside_file.filename):
                inside_data = base64.b64encode(inside_file.read()).decode('utf-8')
                inside_type = inside_file.filename.rsplit('.', 1)[1].lower()
                if inside_type == 'jpg':
                    inside_type = 'jpeg'
                additional_images['inside'] = {'data': inside_data, 'type': inside_type}
    
    # Get ICC profile - either from new upload or existing saved profile
    icc_base64 = None
    
    # Check for new ICC profile upload
    if 'icc_file' in request.files:
        icc_file = request.files['icc_file']
        if icc_file.filename != '' and icc_file.filename.lower().endswith('.icc'):
            # Save the new profile
            saved_filename = save_icc_profile(icc_file)
            if saved_filename:
                # Read it back as base64
                icc_base64 = get_icc_profile_base64(saved_filename)
    
    # If no new upload, check for selected existing profile
    if not icc_base64:
        selected_profile = request.form.get('icc_profile', '')
        if selected_profile:
            icc_base64 = get_icc_profile_base64(selected_profile)
    
    # Build settings from form
    settings = {
        'card_type': card_type,
        'pdf_profile': request.form.get('pdf_profile', 'PDF/X-4'),
        'icc_base64': icc_base64,  # Base64 encoded ICC file
        'add_bleed': request.form.get('add_bleed') == 'true',
        'include_crop_marks': request.form.get('include_crop_marks') == 'true',
        'use_true_black': request.form.get('use_true_black') == 'true',
        'use_cmyk_colors': request.form.get('use_cmyk_colors') == 'true',
        'force_cmyk': request.form.get('force_cmyk') == 'true',
        'image_fit': request.form.get('image_fit', 'cover'),
        'background_color': request.form.get('background_color', '#ffffff'),
        'include_branding': request.form.get('include_branding') == 'true',
        'branding_height': request.form.get('branding_height', '0.375'),
        'branding_logo_size': request.form.get('branding_logo_size', '0.30'),
        'heart_color': request.form.get('heart_color', '#bd2231'),
        'text_color': request.form.get('text_color', '#ffffff'),
        'test_mode': request.form.get('test_mode') == 'true',
    }
    
    # Envelope-specific settings
    if card_type == 'envelope':
        settings['envelope'] = {
            'return_name': request.form.get('return_name', 'JOHN DOE'),
            'return_address': request.form.get('return_address', '123 MAIN STREET\nANYTOWN, ST 12345'),
            'delivery_name': request.form.get('delivery_name', 'JANE SMITH'),
            'delivery_address': request.form.get('delivery_address', '456 OAK AVENUE APT 2B\nSOMEWHERE, ST 67890'),
            'text_color': request.form.get('envelope_text_color', '#000000'),
            'font_family': request.form.get('envelope_font', 'Caveat'),
        }
    
    # Upscale images to meet 300 DPI print requirement
    dpi_warnings = []
    dims = PRINT_DIMENSIONS.get(card_type, PRINT_DIMENSIONS['flat'])
    
    if image_data:
        panel_w = dims['panel_w']
        panel_h = dims['panel_h']
        image_data, image_type, front_dpi = ensure_print_dpi(
            image_data, image_type, panel_w, panel_h
        )
        if front_dpi.get('warning'):
            dpi_warnings.append(f"Front: {front_dpi['warning']}")
    
    if 'back' in additional_images:
        back_img = additional_images['back']
        back_img['data'], back_img['type'], back_dpi = ensure_print_dpi(
            back_img['data'], back_img['type'], dims['panel_w'], dims['panel_h']
        )
        if back_dpi.get('warning'):
            dpi_warnings.append(f"Back: {back_dpi['warning']}")
    
    if 'inside' in additional_images:
        inside_img = additional_images['inside']
        spread_w = dims.get('spread_w', dims['panel_w'])
        spread_h = dims.get('spread_h', dims['panel_h'])
        inside_img['data'], inside_img['type'], inside_dpi = ensure_print_dpi(
            inside_img['data'], inside_img['type'], spread_w, spread_h
        )
        if inside_dpi.get('warning'):
            dpi_warnings.append(f"Inside: {inside_dpi['warning']}")
    
    # Generate HTML
    html_content = generate_html_for_image(image_data, image_type, settings, additional_images)
    
    # Create PDF
    pdf_content, error = create_pdf(html_content, settings, api_key)
    
    if error:
        return jsonify({'error': f'DocRaptor API error: {error}'}), 500
    
    # Save PDF to temp file and return
    pdf_profile_name = settings['pdf_profile'].replace('/', '-') if settings['pdf_profile'] else 'default'
    output_filename = f"{front_image_name}_{pdf_profile_name}.pdf"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    with open(temp_path, 'wb') as f:
        f.write(pdf_content)
    
    response = send_file(
        temp_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=output_filename
    )
    if dpi_warnings:
        response.headers['X-DPI-Warnings'] = ' | '.join(dpi_warnings)
    return response


@app.route('/preview-html', methods=['POST'])
def preview_html():
    """Preview the generated HTML without calling DocRaptor."""
    
    card_type = request.form.get('card_type', 'flat')
    
    # Get uploaded image (optional for envelope)
    image_data = None
    image_type = None
    
    has_image = 'image' in request.files and request.files['image'].filename != ''
    
    if has_image:
        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        image_data = base64.b64encode(file.read()).decode('utf-8')
        image_type = file.filename.rsplit('.', 1)[1].lower()
        if image_type == 'jpg':
            image_type = 'jpeg'
    elif card_type != 'envelope':
        return jsonify({'error': 'No image file provided'}), 400
    
    # Process additional images if provide_all_images is checked
    additional_images = {}
    provide_all = request.form.get('provide_all_images') == 'true'
    
    if provide_all:
        # Get back image
        if 'back_image' in request.files:
            back_file = request.files['back_image']
            if back_file.filename != '' and allowed_file(back_file.filename):
                back_data = base64.b64encode(back_file.read()).decode('utf-8')
                back_type = back_file.filename.rsplit('.', 1)[1].lower()
                if back_type == 'jpg':
                    back_type = 'jpeg'
                additional_images['back'] = {'data': back_data, 'type': back_type}
        
        # Get inside image (only for folded cards)
        if card_type == 'folded' and 'inside_image' in request.files:
            inside_file = request.files['inside_image']
            if inside_file.filename != '' and allowed_file(inside_file.filename):
                inside_data = base64.b64encode(inside_file.read()).decode('utf-8')
                inside_type = inside_file.filename.rsplit('.', 1)[1].lower()
                if inside_type == 'jpg':
                    inside_type = 'jpeg'
                additional_images['inside'] = {'data': inside_data, 'type': inside_type}
    
    # Get ICC profile from selected existing profile
    icc_base64 = None
    selected_profile = request.form.get('icc_profile', '')
    if selected_profile:
        icc_base64 = get_icc_profile_base64(selected_profile)
    
    # Build settings from form
    settings = {
        'card_type': card_type,
        'pdf_profile': request.form.get('pdf_profile', 'PDF/X-4'),
        'icc_base64': icc_base64,
        'add_bleed': request.form.get('add_bleed') == 'true',
        'include_crop_marks': request.form.get('include_crop_marks') == 'true',
        'use_true_black': request.form.get('use_true_black') == 'true',
        'use_cmyk_colors': request.form.get('use_cmyk_colors') == 'true',
        'image_fit': request.form.get('image_fit', 'cover'),
        'background_color': request.form.get('background_color', '#ffffff'),
        'include_branding': request.form.get('include_branding') == 'true',
        'branding_height': request.form.get('branding_height', '0.375'),
        'branding_logo_size': request.form.get('branding_logo_size', '0.30'),
        'heart_color': request.form.get('heart_color', '#bd2231'),
        'text_color': request.form.get('text_color', '#ffffff'),
    }
    
    # Envelope-specific settings
    if card_type == 'envelope':
        settings['envelope'] = {
            'return_name': request.form.get('return_name', 'JOHN DOE'),
            'return_address': request.form.get('return_address', '123 MAIN STREET\nANYTOWN, ST 12345'),
            'delivery_name': request.form.get('delivery_name', 'JANE SMITH'),
            'delivery_address': request.form.get('delivery_address', '456 OAK AVENUE APT 2B\nSOMEWHERE, ST 67890'),
            'text_color': request.form.get('envelope_text_color', '#000000'),
            'font_family': request.form.get('envelope_font', 'Caveat'),
        }
    
    # Upscale images to meet 300 DPI print requirement
    dpi_warnings = []
    dims = PRINT_DIMENSIONS.get(card_type, PRINT_DIMENSIONS['flat'])
    
    if image_data:
        image_data, image_type, front_dpi = ensure_print_dpi(
            image_data, image_type, dims['panel_w'], dims['panel_h']
        )
        if front_dpi.get('warning'):
            dpi_warnings.append(f"Front: {front_dpi['warning']}")
    
    if 'back' in additional_images:
        back_img = additional_images['back']
        back_img['data'], back_img['type'], back_dpi = ensure_print_dpi(
            back_img['data'], back_img['type'], dims['panel_w'], dims['panel_h']
        )
        if back_dpi.get('warning'):
            dpi_warnings.append(f"Back: {back_dpi['warning']}")
    
    if 'inside' in additional_images:
        inside_img = additional_images['inside']
        spread_w = dims.get('spread_w', dims['panel_w'])
        spread_h = dims.get('spread_h', dims['panel_h'])
        inside_img['data'], inside_img['type'], inside_dpi = ensure_print_dpi(
            inside_img['data'], inside_img['type'], spread_w, spread_h
        )
        if inside_dpi.get('warning'):
            dpi_warnings.append(f"Inside: {inside_dpi['warning']}")
    
    # Generate HTML
    html_content = generate_html_for_image(image_data, image_type, settings, additional_images)
    
    result = {'html': html_content}
    if dpi_warnings:
        result['dpi_warnings'] = dpi_warnings
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
