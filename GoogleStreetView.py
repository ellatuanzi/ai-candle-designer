# app.py
import io
import math
import os
from typing import List, Tuple
import base64

import numpy as np
import requests
from PIL import Image, ImageOps, ImageDraw, ImageFont
import streamlit as st

# Gemini
from google import genai
from google.genai import types

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, LETTER, landscape, portrait
from reportlab.lib.units import mm as RL_MM
from reportlab.lib.utils import ImageReader

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="StreetView → Candle Wrap", layout="centered")
st.title("Google Street View → Candle Glass Wrap Generator")
st.caption(
    "Fetch Street View tiles around a point, stitch them into a near-360° panorama, "
    "then output a high-contrast grayscale wrap suitable for translucent paper."
)

# -----------------------------
# SIDEBAR: INPUTS
# -----------------------------
with st.sidebar:
    st.header("API Keys")
    api_key = st.text_input("Google Maps API Key", type="password", help="Enable Street View Static API")
    gemini_key_sidebar = st.text_input("Gemini API Key (for AI generation)", type="password", help="Get from Google AI Studio")
    
    st.header("Google & Location")
    lat = st.number_input("Latitude", value=39.916344, format="%.6f", help="Example: Forbidden City")
    lng = st.number_input("Longitude", value=116.397155, format="%.6f")

    st.header("Sizing Mode")
    sizing_mode = st.radio(
        "Choose how to size the panorama",
        ["Manual pixels", "Physical dimensions (mm → px by DPI)"],
        index=1
    )

    dpi = st.number_input("Print DPI", min_value=72, value=300, step=10)

    if sizing_mode == "Manual pixels":
        out_w = st.number_input("Output width (px)", min_value=600, value=3600, step=100)
        out_h = st.number_input("Output height (px)", min_value=200, value=900, step=50)
    else:
        circumference_mm = st.number_input(
            "Candle wrap visible circumference (mm) — measure with a soft tape",
            min_value=50.0, value=220.0, step=1.0
        )
        visible_h_mm = st.number_input(
            "Candle wrap visible height (mm) — do NOT include bleed",
            min_value=20.0, value=80.0, step=1.0
        )
        # Convert to pixels (overlap & bleed are added later)
        out_w = int(round(circumference_mm * dpi / 25.4))
        out_h = int(round(visible_h_mm * dpi / 25.4))
        st.caption(f"Computed base size: {out_w} × {out_h} px @ {dpi} DPI")

    st.header("Sampling & Stitching")
    fov_deg = st.slider("FOV per tile (°)", 15, 45, 26, help="Smaller FOV = less distortion, more requests.")
    overlap_pct = st.slider("Overlap ratio (%)", 10, 60, 35)
    pitch_deg = st.slider("Pitch (°)", -20, 20, 0)
    source_outdoor = st.checkbox("Prefer outdoor scenes (source=outdoor)", value=True)
    radius_m = st.number_input("Search radius (m)", min_value=0, value=50)

    st.header("Wrap Finishing")
    add_overlap_mm = st.number_input("Glue overlap at right edge (mm)", min_value=0.0, value=3.0, step=0.5)
    add_bleed_mm = st.number_input("Top & bottom bleed (each, mm)", min_value=0.0, value=2.0, step=0.5)

    st.header("Style")
    style = st.selectbox("Grayscale mode", ["Normal grayscale", "High-contrast threshold (line-art style)"])
    threshold = st.slider("Threshold (for line-art)", 20, 235, 160)
    seam_guides = st.checkbox("Show cut/overlap guides on PNG", value=True)
    shift_deg = st.slider("Horizontal rotation offset (°)", 0, 359, 0,
                          help="Rotate start heading so your 'main view' lands at the front seam.")

    st.header("Advanced")
    tile_px = st.selectbox("Tile size to fetch", ["640x640", "640x480", "800x800 (high quota)", "1024x1024 (high quota)"])

# -----------------------------
# HELPERS
# -----------------------------
def parse_size(opt: str) -> Tuple[int, int]:
    w, h = opt.split("x")
    return int(w), int(h)

def deg_step(fov: float, overlap_ratio: float) -> float:
    return fov * (1.0 - overlap_ratio)

def headings_for_full_circle(step_deg: float, shift: int = 0) -> List[int]:
    """Generate headings around 360 degrees in sequential order."""
    n = max(1, int(math.ceil(360.0 / step_deg)))
    # Generate headings sequentially without wrapping
    # Start from shift and go around, keeping values in order
    vals = []
    for i in range(n):
        heading = (shift + i * step_deg) % 360
        vals.append(int(heading))
    return vals

def check_streetview_metadata(lat, lng, api_key, radius=50, source_outdoor=True):
    """Check if Street View is available at a location."""
    base = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "key": api_key
    }
    if source_outdoor:
        params["source"] = "outdoor"
    r = requests.get(base, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data

def fetch_streetview(lat, lng, heading, pitch, fov, size_wh, api_key, source_outdoor=True, radius=50, pano_id=None):
    base = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": f"{size_wh[0]}x{size_wh[1]}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": api_key
    }
    
    # Use pano_id if provided, otherwise use location
    if pano_id:
        params["pano"] = pano_id
    else:
        params["location"] = f"{lat},{lng}"
        params["radius"] = radius
        if source_outdoor:
            params["source"] = "outdoor"
    
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img

def stitch_panorama(tiles: List[Image.Image], slice_w: int, target_h: int, headings: List[int] = None) -> Image.Image:
    """Crop the vertical center slice of each tile and join side-by-side."""
    slices = []
    for i, im in enumerate(tiles):
        w, h = im.size
        x0 = (w - slice_w) // 2
        crop = im.crop((x0, 0, x0 + slice_w, h))
        
        # Add heading text overlay for debugging
        if headings and len(headings) > i:
            draw = ImageDraw.Draw(crop)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
            text = f"{headings[i]}°"
            draw.text((5, 5), text, fill="red", font=font)
        
        slices.append(crop)
    
    pano = Image.new("RGB", (slice_w * len(slices), tiles[0].height))
    x = 0
    for i, s in enumerate(slices):
        pano.paste(s, (x, 0))
        x += s.width
    
    if pano.height != target_h:
        pano = pano.resize((int(pano.width * (target_h / pano.height)), target_h), Image.BICUBIC)
    return pano

def to_grayscale_or_threshold(img: Image.Image, mode: str, thresh: int) -> Image.Image:
    g = ImageOps.grayscale(img)
    if "High-contrast" in mode:
        return g.point(lambda p: 255 if p >= thresh else 0, mode='1').convert("L")
    return g

def add_wrap_and_guides(img: Image.Image, dpi: int, overlap_mm: float, bleed_mm: float, show_guides=True):
    """Add right-edge overlap and top/bottom bleed (in mm). Draw basic guides if requested."""
    px_per_mm = dpi / 25.4
    ow = int(round(overlap_mm * px_per_mm))
    bh = int(round(bleed_mm * px_per_mm))
    W, H = img.size
    canvas = Image.new("L", (W + ow, H + 2 * bh), color=255)
    canvas.paste(img, (0, bh))
    if show_guides:
        draw = ImageDraw.Draw(canvas)
        # Top/Bottom cut lines (bleed boundary)
        draw.line([(0, bh), (canvas.width, bh)], fill=0, width=1)
        draw.line([(0, canvas.height - bh - 1), (canvas.width, canvas.height - bh - 1)], fill=0, width=1)
        # Right overlap boundary
        draw.line([(W, 0), (W, canvas.height)], fill=0, width=1)
        # Corner ticks
        for x in (10, canvas.width - 10):
            draw.line([(x, 10), (x, 40)], fill=0, width=1)
        for y in (10, canvas.height - 10):
            draw.line([(10, y), (40, y)], fill=0, width=1)
        font = ImageFont.load_default()
        label = f"Print @ {dpi} DPI | Overlap {overlap_mm} mm | Bleed {bleed_mm} mm"
        draw.text((12, 12), label, fill=0, font=font)
    return canvas

def px_to_mm(px, dpi):
    return round(px * 25.4 / dpi, 1)

# ---------- PDF helpers ----------
def _points_from_px(px: int, dpi: int) -> float:
    # 1 in = 72 points; px / dpi = inches ⇒ *72 for points
    return (px / dpi) * 72.0

def draw_crop_marks(c: canvas.Canvas, x0: float, y0: float, x1: float, y1: float,
                    mark_len_mm: float = 5.0, offset_mm: float = 1.5, line_w: float = 0.4):
    """Draw crop marks around rectangle [x0,y0,x1,y1] in PDF points."""
    ml = mark_len_mm * RL_MM
    off = offset_mm * RL_MM
    c.setLineWidth(line_w)
    # Top-left
    c.line(x0 - off - ml, y1 + off, x0 - off, y1 + off)  # horizontal
    c.line(x0 - off, y1 + off, x0 - off, y1 + off + ml)  # vertical
    # Top-right
    c.line(x1 + off, y1 + off, x1 + off + ml, y1 + off)
    c.line(x1 + off, y1 + off, x1 + off, y1 + off + ml)
    # Bottom-left
    c.line(x0 - off - ml, y0 - off, x0 - off, y0 - off)
    c.line(x0 - off, y0 - off - ml, x0 - off, y0 - off)
    # Bottom-right
    c.line(x1 + off, y0 - off, x1 + off + ml, y0 - off)
    c.line(x1 + off, y0 - off - ml, x1 + off, y0 - off)

def export_pdf_exact(final_img: Image.Image, dpi: int) -> bytes:
    """Create a PDF whose page size exactly matches the image size at given DPI."""
    w_px, h_px = final_img.size
    w_pt = _points_from_px(w_px, dpi)
    h_pt = _points_from_px(h_px, dpi)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(w_pt, h_pt))
    # Place image covering the page exactly
    img_buf = io.BytesIO()
    final_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 0, 0, width=w_pt, height=h_pt, preserveAspectRatio=False, mask='auto')
    # Crop marks just outside (they may clip on exact page—still useful at edges)
    draw_crop_marks(c, 0, 0, w_pt, h_pt)
    c.showPage()
    c.save()
    return buf.getvalue()

def export_pdf_fit(final_img: Image.Image, dpi: int, paper: str = "A4", margins_mm: float = 10.0) -> bytes:
    """Create a PDF that fits the image inside A4/Letter with margins (may scale)."""
    if paper == "A4":
        base = A4
    else:
        base = LETTER
    # Choose orientation based on image aspect
    w_px, h_px = final_img.size
    if w_px >= h_px:
        page_w, page_h = landscape(base)
    else:
        page_w, page_h = portrait(base)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    # Available area
    margin = margins_mm * RL_MM
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin

    # Image size in points at 1:1
    img_w_pt = _points_from_px(w_px, dpi)
    img_h_pt = _points_from_px(h_px, dpi)

    scale = min(avail_w / img_w_pt, avail_h / img_h_pt, 1.0)
    draw_w = img_w_pt * scale
    draw_h = img_h_pt * scale
    x = (page_w - draw_w) / 2.0
    y = (page_h - draw_h) / 2.0

    # Render image
    img_buf = io.BytesIO()
    final_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), x, y, width=draw_w, height=draw_h,
                preserveAspectRatio=False, mask='auto')

    # Crop marks on the image box
    draw_crop_marks(c, x, y, x + draw_w, y + draw_h)
    c.showPage()
    c.save()
    return buf.getvalue()

# -----------------------------
# MAIN ACTION
# -----------------------------
# Initialize session state for tiles
if 'tiles' not in st.session_state:
    st.session_state.tiles = None
    st.session_state.fetched_headings = None

left, right = st.columns(2)
with left:
    run = st.button("Generate Wrap", type="primary")
with right:
    st.write("")

if run:
    if not api_key:
        st.error("Please provide a valid Google Maps API key.")
        st.stop()

    # First, check if Street View is available at this location
    with st.spinner("Checking Street View availability..."):
        try:
            metadata = check_streetview_metadata(lat, lng, api_key, radius=radius_m, source_outdoor=source_outdoor)
            if metadata.get("status") != "OK":
                st.error(f"❌ No Street View imagery found at this location.")
                st.warning(f"**Status:** {metadata.get('status', 'UNKNOWN')}")
                st.info(f"""
                **Suggestions:**
                - Increase the search radius (currently {radius_m}m)
                - Try different coordinates
                - Disable 'Prefer outdoor scenes' if you're searching indoors
                - Check that Street View is available in this area on Google Maps
                """)
                if metadata.get("status") == "ZERO_RESULTS":
                    st.caption("ZERO_RESULTS means no Street View coverage within the search radius.")
                st.stop()
            else:
                actual_lat = metadata.get("location", {}).get("lat", lat)
                actual_lng = metadata.get("location", {}).get("lng", lng)
                pano_id = metadata.get("pano_id", "N/A")
                st.success(f"✅ Street View found! Pano ID: {pano_id}")
                st.info(f"📍 Actual location: {actual_lat:.6f}, {actual_lng:.6f}")
        except Exception as e:
            st.error(f"Error checking Street View metadata: {e}")
            st.stop()

    with st.spinner("Fetching Street View tiles and stitching..."):
        tile_w, tile_h = parse_size(tile_px)
        step = deg_step(fov_deg, overlap_pct / 100.0)
        headings = headings_for_full_circle(step, shift_deg)

        st.info(f"🔄 Fetching {len(headings)} tiles from pano {pano_id}")

        # Store tiles with their headings to maintain order
        tile_data = []
        errors = []
        
        for hdg in headings:
            try:
                # Use pano_id to ensure all tiles come from the same panorama
                img = fetch_streetview(lat, lng, hdg, pitch_deg, fov_deg,
                                       (tile_w, tile_h), api_key,
                                       source_outdoor=source_outdoor, radius=radius_m,
                                       pano_id=pano_id)
                tile_data.append((hdg, img))
            except Exception as e:
                errors.append((hdg, str(e)))

        if errors:
            st.warning(f"⚠️ Failed to fetch {len(errors)} tiles at headings: {[e[0] for e in errors]}")

        if not tile_data:
            st.error("No Street View image found. Try adjusting coordinates or radius.")
            st.stop()

        # Keep tiles in the order they were fetched (sequential around 360°)
        # DO NOT sort - they are already in correct panoramic order!
        tiles = [img for hdg, img in tile_data]
        fetched_headings = [hdg for hdg, img in tile_data]
        
        # Store in session state so they persist across reruns
        st.session_state.tiles = tiles
        st.session_state.fetched_headings = fetched_headings
        
        st.success(f"✅ Successfully fetched {len(tiles)} tiles")
        st.info(f"📊 Tiles in panoramic order: {fetched_headings}")

        # Show preview of ALL tiles to verify order
        st.write("**Preview of all tiles (in order):**")
        num_cols = min(8, len(tiles))
        for row_start in range(0, len(tiles), num_cols):
            preview_cols = st.columns(num_cols)
            for i, col in enumerate(preview_cols):
                tile_idx = row_start + i
                if tile_idx < len(tiles):
                    with col:
                        st.image(tiles[tile_idx], caption=f"#{tile_idx}: {fetched_headings[tile_idx]}°", use_container_width=True)

        # Compute slice width so total ≈ target width
        est_slice_w = max(2, int(math.ceil(out_w / len(tiles))))
        slice_w = min(est_slice_w, tile_w)


        st.success("✅ Tiles fetched successfully! Use the features below to create your image.")

# ===== GEMINI AI IMAGE GENERATION FEATURE =====
if st.session_state.tiles is not None and len(st.session_state.tiles) > 0:
    st.divider()
    st.subheader("🤖 AI-Powered Image Generation (Gemini)")
    st.write("Let Gemini analyze the Street View tiles and generate a custom candle wrap design")
    
    tiles = st.session_state.tiles
    fetched_headings = st.session_state.fetched_headings
    
    # User prompt for what they want
    user_prompt = st.text_area(
        "Describe what kind of candle wrap you want Gemini to create",
        value="Create a beautiful candle wrap design using the best views from these Street View images. Combine them artistically into a cohesive panoramic design suitable for wrapping around a candle.",
        height=100
    )
    
    if st.button("✨ Generate with Gemini AI"):
        # Set environment variable from sidebar if provided
        if gemini_key_sidebar:
            os.environ["GEMINI_API_KEY"] = gemini_key_sidebar
        
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if not gemini_api_key:
            st.error("❌ Please enter your Gemini API key in the sidebar")
        else:
            with st.spinner("Sending images to Gemini and generating design..."):
                try:
                    # Initialize Gemini client
                    client = genai.Client(api_key=gemini_api_key)
                    
                    # Convert tiles to base64 for sending to Gemini
                    image_parts = []
                    for i, tile in enumerate(tiles[:8]):  # Send first 8 tiles to avoid token limit
                        img_buffer = io.BytesIO()
                        tile.save(img_buffer, format='JPEG', quality=85)
                        img_buffer.seek(0)
                        img_bytes = img_buffer.read()
                        
                        image_parts.append(
                            types.Part.from_bytes(
                                data=img_bytes,
                                mime_type="image/jpeg"
                            )
                        )
                    
                    # Calculate physical dimensions
                    width_mm = round(out_w * 25.4 / dpi, 1)
                    height_mm = round(out_h * 25.4 / dpi, 1)
                    aspect_ratio = round(out_w / out_h, 2)
                    
                    # Create prompt with context and dimensions
                    full_prompt = f"""You have been provided with {len(image_parts)} Street View images from different angles (headings: {fetched_headings[:8]}).

TARGET DIMENSIONS (CRITICAL - MUST MATCH):
- Width: {out_w} pixels ({width_mm} mm)
- Height: {out_h} pixels ({height_mm} mm)
- Resolution: {dpi} DPI
- Aspect Ratio: {aspect_ratio}:1 (wide horizontal panoramic format)

IMPORTANT: The generated image MUST be in a wide horizontal panoramic format with aspect ratio approximately {aspect_ratio}:1. This is for wrapping around a cylindrical candle.

{user_prompt}

Please generate a horizontal panoramic image that EXACTLY matches the aspect ratio of {aspect_ratio}:1 and would work well as a candle wrap. The image should be visually cohesive, wide and horizontal, and suitable for printing on translucent paper at {dpi} DPI."""
                    
                    # Configure for image generation
                    generate_content_config = types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    )
                    
                    # Send to Gemini for image generation using streaming
                    final_bytes = None
                    found_image = False
                    
                    for chunk in client.models.generate_content_stream(
                        model='gemini-2.5-flash-image',
                        contents=[
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=full_prompt)] + image_parts
                            )
                        ],
                        config=generate_content_config
                    ):
                        if (
                            chunk.candidates is None
                            or chunk.candidates[0].content is None
                            or chunk.candidates[0].content.parts is None
                        ):
                            continue

                        part = chunk.candidates[0].content.parts[0]

                        if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                            data_buffer = part.inline_data.data
                            generated_img = Image.open(io.BytesIO(data_buffer))
                            generated_img.load()  # Force load to validate
                            
                            final_bytes = data_buffer
                            found_image = True
                            break
                    
                    if found_image and final_bytes:
                        st.success("✅ Gemini generated your candle wrap design!")
                        
                        # Show dimensions info
                        gen_w, gen_h = generated_img.size
                        gen_ratio = round(gen_w / gen_h, 2)
                        target_ratio = round(out_w / out_h, 2)
                        
                        st.info(f"📐 Generated: {gen_w}×{gen_h}px (ratio {gen_ratio}:1) | Target: {out_w}×{out_h}px (ratio {target_ratio}:1)")
                        
                        # Resize to match target dimensions if needed
                        if gen_w != out_w or gen_h != out_h:
                            st.warning(f"⚠️ Resizing generated image from {gen_w}×{gen_h} to match target {out_w}×{out_h}")
                            generated_img = generated_img.resize((out_w, out_h), Image.LANCZOS)
                            
                            # Update bytes for download
                            img_buf = io.BytesIO()
                            generated_img.save(img_buf, format="PNG")
                            final_bytes = img_buf.getvalue()
                        
                        st.image(generated_img, caption=f"AI-Generated Candle Wrap ({out_w}×{out_h}px)", use_container_width=True)
                        
                        # Download button
                        st.download_button(
                            label="💾 Download AI-Generated Wrap",
                            data=final_bytes,
                            file_name="gemini_candle_wrap.png",
                            mime="image/png"
                        )
                    else:
                        st.error("❌ Model did not return an image. Please try again or adjust your prompt.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating with Gemini: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

st.markdown("""
**Notes & Compliance**  
• Please comply with Google Maps Platform Terms of Service — keep watermarks/copyright.  
• The 360° effect is approximated by sampling many headings with a small FOV and stitching center slices.  
• For best print results on translucent paper (vellum), use high contrast and 300 DPI.  
""")
