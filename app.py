"""
DocRaptor PDF Output Tester
A Flask application to test different PDF output settings using DocRaptor API.
"""

import os
import base64
import tempfile
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import docraptor

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

# Simulated back panel content
BACK_PANEL_CONTENT = """
<div class="back-content">
    <div class="logo-placeholder">✦</div>
    <h2>Premium Greeting Card</h2>
    <p class="tagline">Crafted with care, delivered with love</p>
    <div class="details">
        <p>Made in USA</p>
        <p>Recycled Paper</p>
        <p>www.example.com</p>
    </div>
    <div class="barcode-placeholder">
        <div class="barcode-lines"></div>
        <span>1234567890</span>
    </div>
</div>
"""

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
        
        /* Back panel styles */
        .back-content {
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
        
        .back-content .logo-placeholder {
            font-size: 48pt;
            color: #c4a052;
            margin-bottom: 0.2in;
        }
        
        .back-content h2 {
            font-size: 14pt;
            font-weight: normal;
            letter-spacing: 2pt;
            text-transform: uppercase;
            margin-bottom: 0.1in;
        }
        
        .back-content .tagline {
            font-size: 10pt;
            font-style: italic;
            color: #666;
            margin-bottom: 0.3in;
        }
        
        .back-content .details {
            font-size: 8pt;
            color: #888;
            line-height: 1.6;
            margin-bottom: 0.3in;
        }
        
        .back-content .barcode-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .back-content .barcode-lines {
            width: 1.2in;
            height: 0.4in;
            background: repeating-linear-gradient(
                90deg,
                #000 0px,
                #000 2px,
                #fff 2px,
                #fff 4px,
                #000 4px,
                #000 5px,
                #fff 5px,
                #fff 8px
            );
            margin-bottom: 0.05in;
        }
        
        .back-content .barcode-placeholder span {
            font-family: 'Courier New', monospace;
            font-size: 8pt;
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
    card_width = 4.75  # inches
    card_height = 6.75  # inches
    bleed = 0.125 if settings.get('add_bleed', False) else 0  # 1/8 inch
    
    # Total dimensions with bleed
    total_width = card_width + (bleed * 2)
    total_height = card_height + (bleed * 2)
    
    prince_pdf_block = get_prince_pdf_css(settings)
    fit_mode = settings.get('image_fit', 'cover')
    bg_color = settings.get('background_color', '#ffffff')
    
    # Marks: crop only (no cross/registration marks)
    marks = 'crop' if bleed > 0 else 'none'
    
    # Determine back panel content
    if back_image_data and back_image_type:
        back_content = f'<img class="image" src="data:image/{back_image_type};base64,{back_image_data}" alt="Card Back">'
    else:
        back_content = BACK_PANEL_CONTENT
    
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
        
        .back-page-content {{
            position: absolute;
            top: -{bleed}in;
            left: -{bleed}in;
            width: {total_width}in;
            height: {total_height}in;
            background-color: {bg_color};
            display: flex;
            align-items: center;
            justify-content: center;
        }}
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
        <div class="back-page-content">
            {back_content}
        </div>
    </div>
</body>
</html>"""
    
    return html


def generate_folded_card_html(image_data, image_type, settings, inside_image_data=None, inside_image_type=None, back_image_data=None, back_image_type=None):
    """Generate HTML for folded card (2 spreads: outside and inside)."""
    
    # Single panel dimensions (trim size)
    panel_width = 4.75  # inches
    panel_height = 6.75  # inches
    bleed = 0.125 if settings.get('add_bleed', False) else 0  # 1/8 inch
    
    # Spread dimensions (2 panels side by side) - this is the TRIM size
    spread_width = panel_width * 2  # 9.5 inches
    spread_height = panel_height  # 6.75 inches
    
    # Total dimensions with bleed (content extends beyond trim)
    total_spread_width = spread_width + (bleed * 2)  # 9.75 inches with bleed
    total_spread_height = spread_height + (bleed * 2)  # 7 inches with bleed
    
    prince_pdf_block = get_prince_pdf_css(settings)
    fit_mode = settings.get('image_fit', 'cover')
    bg_color = settings.get('background_color', '#ffffff')
    
    # Marks: crop only (no cross/registration marks)
    marks = 'crop' if bleed > 0 else 'none'
    
    # Determine back panel content (Panel 4)
    if back_image_data and back_image_type:
        back_panel_content = f'<img class="image" src="data:image/{back_image_type};base64,{back_image_data}" alt="Back Cover">'
    else:
        back_panel_content = PANEL_4_CONTENT
    
    # Determine inside panel content (spans both Panel 2 and Panel 3)
    if inside_image_data and inside_image_type:
        # Single image spans both inside panels
        inside_left_content = f'<img class="image" src="data:image/{inside_image_type};base64,{inside_image_data}" alt="Inside Left" style="object-position: right center;">'
        inside_right_content = f'<img class="image" src="data:image/{inside_image_type};base64,{inside_image_data}" alt="Inside Right" style="object-position: left center;">'
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
    
    # Get uploaded image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Read and encode the front image
    image_data = base64.b64encode(file.read()).decode('utf-8')
    image_type = file.filename.rsplit('.', 1)[1].lower()
    if image_type == 'jpg':
        image_type = 'jpeg'
    
    # Process additional images if provide_all_images is checked
    additional_images = {}
    provide_all = request.form.get('provide_all_images') == 'true'
    card_type = request.form.get('card_type', 'flat')
    
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
        'use_true_black': request.form.get('use_true_black') == 'true',
        'use_cmyk_colors': request.form.get('use_cmyk_colors') == 'true',
        'force_cmyk': request.form.get('force_cmyk') == 'true',
        'image_fit': request.form.get('image_fit', 'cover'),
        'background_color': request.form.get('background_color', '#ffffff'),
        'test_mode': request.form.get('test_mode') == 'true',
    }
    
    # Generate HTML
    html_content = generate_html_for_image(image_data, image_type, settings, additional_images)
    
    # Create PDF
    pdf_content, error = create_pdf(html_content, settings, api_key)
    
    if error:
        return jsonify({'error': f'DocRaptor API error: {error}'}), 500
    
    # Save PDF to temp file and return
    output_filename = f"output_{settings['pdf_profile'].replace('/', '-') if settings['pdf_profile'] else 'default'}.pdf"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    with open(temp_path, 'wb') as f:
        f.write(pdf_content)
    
    return send_file(
        temp_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=output_filename
    )


@app.route('/preview-html', methods=['POST'])
def preview_html():
    """Preview the generated HTML without calling DocRaptor."""
    
    # Get uploaded image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Read and encode the front image
    image_data = base64.b64encode(file.read()).decode('utf-8')
    image_type = file.filename.rsplit('.', 1)[1].lower()
    if image_type == 'jpg':
        image_type = 'jpeg'
    
    # Process additional images if provide_all_images is checked
    additional_images = {}
    provide_all = request.form.get('provide_all_images') == 'true'
    card_type = request.form.get('card_type', 'flat')
    
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
        'use_true_black': request.form.get('use_true_black') == 'true',
        'use_cmyk_colors': request.form.get('use_cmyk_colors') == 'true',
        'image_fit': request.form.get('image_fit', 'cover'),
        'background_color': request.form.get('background_color', '#ffffff'),
    }
    
    # Generate HTML
    html_content = generate_html_for_image(image_data, image_type, settings, additional_images)
    
    return jsonify({'html': html_content})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
