import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load Stable Diffusion model for image generation
@st.cache_resource
def load_sd_model():
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16"
    )
    return model.to("cuda")

# Load CLIP model to compute prompt-image relevance
@st.cache_resource
def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, processor

# Compute raw CLIP similarity score (no softmax)
def get_clip_score(image, prompt, model, processor):
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits_per_image[0][0].item()
    return round(score, 2)  # raw cosine similarity score

# Normalize score (optional) to percentage scale
def normalize_clip_score(score):
    # Empirical range: 15 to 30 → map to 0 to 100%
    return round(min(max((score - 15) * 5, 0), 100), 2)

# Load models
sd_model = load_sd_model()
clip_model, clip_processor = load_clip_model()

# Streamlit UI
st.title(" Text-to-Image Generator")
st.write("Type a prompt, generate an image, and see how well it matches!")

# Prompt input
prompt = st.text_input("Enter your prompt (e.g., 'A tiger in space')")

if st.button("Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image"):
            # Generate image
            image = sd_model(prompt).images[0]
            # Get CLIP similarity score
            raw_score = get_clip_score(image, prompt, clip_model, clip_processor)
            percent_score = normalize_clip_score(raw_score)

            # Display image and scores
            st.image(image, caption=prompt, use_container_width=True)
            st.markdown(f"** Relevance Score:** `{percent_score}%`")
            st.markdown(f"** Raw CLIP Score:** `{raw_score}`")

