import os
import io

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image


# ========= 核心生成函数：沿用“这版 work”的逻辑 =========
def generate_image_bytes(prompt: str) -> bytes:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please input API key in sidebar.")

    client = genai.Client(api_key=api_key)

    model = "gemini-2.5-flash-image"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )

    final_bytes = None
    collected_text = []

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
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

            img = Image.open(io.BytesIO(data_buffer))
            img.load()

            final_bytes = data_buffer

        elif getattr(part, "text", None):
            collected_text.append(part.text)

    if final_bytes is None:
        if collected_text:
            raise RuntimeError(
                "Model did not return an image. Text output:\n" + "\n".join(collected_text)
            )
        raise RuntimeError("Model did not return any image or text.")

    return final_bytes


# ========= Streamlit UI 部分 =========
st.set_page_config(
    page_title="Candle Design Generator",
    page_icon="🕯️",
    layout="wide",
)

st.title("🕯️ Candle Design Generator (Gemini 2.5 Flash Image)")
st.markdown(
    "Generate **realistic, achievable candle design concepts** using Gemini 2.5 Flash Image.\n"
    "This version uses the stable logic you've already verified."
)

with st.sidebar:
    st.header("⚙️ Configuration")
    sidebar_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Server key from Google AI Studio.",
    )

    st.markdown("---")
    st.markdown(
        "1. Enter API key\n"
        "2. Enter candle design\n"
        "3. Click **Generate Candle Image**"
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Candle Design Input")

    candle_description = st.text_area(
        "Describe your candle design",
        height=150,
        placeholder=(
            "e.g., A multi-layer pillar candle, each layer a different natural wax color, "
            "with dried flowers and herbs embedded realistically inside the wax."
        ),
    )

    keywords = st.text_input(
        "Additional style keywords (optional)",
        placeholder="e.g., botanical, gradient, handmade, rustic",
    )

with col2:
    st.header("🎨 Generated Candle Image")
    status_placeholder = st.empty()
    image_placeholder = st.empty()
    prompt_debug_placeholder = st.empty()

generate_button = st.button(
    "✨ Generate Candle Image",
    type="primary",
    use_container_width=True,     # fix
)


if generate_button:
    if sidebar_key:
        os.environ["GEMINI_API_KEY"] = sidebar_key

    if not os.environ.get("GEMINI_API_KEY"):
        status_placeholder.error("❌ No API key provided.")
    elif not (candle_description or "").strip():
        status_placeholder.error("❌ Please describe your candle design first.")
    else:
        full_description = candle_description.strip()
        if keywords and keywords.strip():
            full_description += f". Style: {keywords.strip()}"

        prompt = (
            "A realistic, achievable candle design illustration. "
            "Handmade multi-layer candle, each layer in a different natural wax color. "
            "Wax textures should look real and practical: slightly uneven surfaces and subtle translucency. "
            "Decorations such as dried flowers or herbs should be embedded realistically—"
            "partially within the wax, not floating. "
            "Show clear layer boundaries so a candle maker can replicate the design. "
            "Natural soft lighting, not studio photography. "
            f"User design details: {full_description}."
        )

        try:
            with st.spinner("Generating candle image..."):
                png_bytes = generate_image_bytes(prompt)

            img = Image.open(io.BytesIO(png_bytes))
            img.load()

            status_placeholder.success("✅ Candle image generated!")

            image_placeholder.image(
                img,
                caption="Your Candle Design",
                use_container_width=True,    # fix
            )

            st.download_button(
                "📥 Download Candle Image",
                data=png_bytes,
                file_name="candle_design.png",
                mime="image/png",
                use_container_width=True,     # fix
            )

            with prompt_debug_placeholder.expander("📝 View Prompt Used"):
                st.code(prompt)

        except Exception as e:
            status_placeholder.error(f"❌ Error: {e}")
