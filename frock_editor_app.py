import streamlit as st
import base64
import os
import mimetypes
import json
import re  # Add this import for regex
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please check your environment variables.")
    
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    st.error("Google API key not found. Please check your environment variables.")
    
# ==== Setup Clients ====
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)
genai_client = genai.Client(api_key=google_key)

# ==== Utility Functions ====

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)

def identify_intent_and_rephrase(user_input):
    system_prompt = """
You are an intent detection and prompt rephrasing system for an image editing assistant. The assistant helps modify images of frocks. It only supports:
1. Sleeve edits (like making sleeves full length, making sleeveless, changing sleeve styles)
2. Frock color changes

Your job:
- Detect if the user's request is about sleeves, color, or other.
- Rephrase the request into a precise image editing prompt. 
- For 'sleeve', match to one of the sleeve types below and return the corresponding base prompt:
    a. Full-length sleeves
    b. Sleeveless
    c. Custom (e.g., cap sleeves, bell sleeves, etc.)
- For 'color', extract the color user wants and place it in the color base prompt.
- For 'other', return prompt as "none".

Sleeve base prompts:
- Full-length sleeves:
 "Extend the short puff sleeves of the frock to full-length sleeves while keeping the same maroon velvet fabric and intricate golden embroidered design pattern matching the bodice style, keeping everything else in the image completely unchanged — including alignment, size, background, and colors. Use the following precise bounding boxes for sleeve inpainting: Left sleeve: (90, 50, 170, 130) Right sleeve: (290, 60, 360, 130). Maintain the original style and fabric look of the frock. Ensure the edit is seamless and visually consistent with the rest of the garment. Do not modify any other part of the image."

 - Sleeveless:
 "Remove the short puff sleeves of the frock while maintaining the same maroon velvet fabric and intricate golden embroidered design pattern on the bodice. Keep everything else in the image completely unchanged — including alignment, size, background, and colors.  Use the following precise bounding boxes for sleeve inpainting: Left sleeve: (90, 50, 170, 130) Right sleeve: (290, 60, 360, 130). Maintain the original style and fabric look of the frock. Ensure the edit is seamless and visually consistent with the rest of the garment. Do not modify any other part of the image."

 - Custom sleeve styles:
 "Change the sleeves of the frock to <custom style> sleeves while keeping the same maroon velvet fabric and intricate golden embroidered design pattern matching the bodice style. Keep everything else in the image completely unchanged — including alignment, size, background, and colors.  Use the following precise bounding boxes for sleeve inpainting: Left sleeve: (90, 50, 170, 130) Right sleeve: (290, 60, 360, 130). Maintain the original style and fabric look of the frock. Ensure the edit is seamless and visually consistent with the rest of the garment. Do not modify any other part of the image."

Color base prompt:
"Change the color of the frock to <color>, keeping the structure, embroidery design, sleeve pattern, skirt layout, fabric texture, and overall alignment exactly the same. Do not alter any other visual element except the color."


Respond in JSON format like this:
{
  "intent": "sleeve" | "color" | "other",
  "prompt": "the edited prompt to send to the image model"
}
""".strip()

    response = openai_client.chat.completions.create(
        extra_body={},
        model="openai/gpt-3.5-turbo-0613",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return json.loads(response.choices[0].message.content)

def generate_image(prompt, image_path, suffix):
    model = "gemini-2.0-flash-exp-image-generation"
    base64_image = encode_image_to_base64(image_path)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type="image/jpeg", data=base64.b64decode(base64_image)),
                types.Part.from_text(text=prompt)
            ],
        )
    ]

    config = types.GenerateContentConfig(
        response_modalities=["image", "text"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE")
        ],
        response_mime_type="text/plain",
    )

    for chunk in genai_client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.candidates and chunk.candidates[0].content.parts[0].inline_data:
            file_name = f"edited_image_{suffix}"
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            ext = mimetypes.guess_extension(inline_data.mime_type)
            file_path = f"{file_name}{ext}"
            save_binary_file(file_path, inline_data.data)
            return file_path
    return None

def compare_images(original_path, edited_path):
    # Verify files exist
    if not os.path.exists(original_path):
        print(f"Error: Original file '{original_path}' not found")
        return 0.0
    if not os.path.exists(edited_path):
        print(f"Error: Edited file '{edited_path}' not found")
        return 0.0
    
    # Initialize client
    try:
        client = genai.Client(api_key="AIzaSyBtXKUP0s-fq9IqBwqmjBppbQZ2BiVyZYI")
        model = "gemini-2.0-flash-exp-image-generation"
    except Exception as e:
        print(f"Error initializing genai client: {e}")
        return 0.0

    # Encode images
    try:
        original_base64 = encode_image_to_base64(original_path)
        edited_base64 = encode_image_to_base64(edited_path)
    except Exception as e:
        print(f"Error encoding images: {e}")
        return 0.0

    # Create prompt
    prompt = """You are a precise image comparison tool. Analyze both images showing a red velvet children's frock with gold embroidery. The second image is an edited version where ONLY the sleeves should be modified.

Evaluate and score (0-100) how well the edited image preserves the original's key elements, based on:

1. Background preservation: Is the background exactly the same? (20 points)
2. Dress structure integrity: Are the following preserved exactly as in original? (50 points total)
   - Bodice presence: Is the bodice part of the dress present at all? (15 points)
   - Bodice design: Is the bodice design maintained AND are the gold embroidery patterns preserved in their original design exactly as in the original image? (15 points)
   - Skirt shape, pleating and length (10 points)
   - Gold embroidery on skirt hem (5 points)
   - Overall fabric texture and appearance (5 points)
3. Color accuracy: Is the red velvet color consistent with original? (15 points)
4. Alignment and positioning: Is the dress positioned identically? (15 points)

IGNORE sleeve changes — these are expected.

RESPONSE INSTRUCTIONS:
- Do not explain your answer.
- Do not include any comments, breakdown, or extra text.
- Respond with one line only in this format:

FINAL_SCORE: [number]
"""

    # Create content
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type="image/jpeg", data=base64.b64decode(original_base64)),
                types.Part.from_bytes(mime_type="image/jpeg", data=base64.b64decode(edited_base64)),
                types.Part.from_text(text=prompt)
            ]
        )
    ]

    # Configure request
    config = types.GenerateContentConfig(
        response_modalities=["text"],
        safety_settings=[],
        response_mime_type="text/plain",
    )

    # Make API request
    try:
        response = client.models.generate_content(model=model, contents=contents, config=config)
        
        # Get text response
        if response.candidates and response.candidates[0].content.parts:
            text_response = response.candidates[0].content.parts[0].text
            
            # Extract number using regex
            number_match = re.search(r'(\d+(?:\.\d+)?)', text_response)
            if number_match:
                score = float(number_match.group(1))
                return score
            else:
                print("Could not extract a numeric score from response")
                return 0.0
        else:
            print("No valid response from model")
            return 0.0
    except Exception as e:
        print(f"Error in API request: {e}")
        return 0.0

# ==== Streamlit App ====

st.set_page_config(page_title="Frock Image Editor", layout="centered")
st.title("👗 Swakriti AI Fashion Designer")

original_image_path = "frockonly.png"
image_container = st.empty()

if os.path.exists(original_image_path):
    image_container.image(original_image_path, caption="Frock Image", width=400)

user_input = st.text_input("Describe the change you want in the frock image (e.g., make sleeves full, change frock to blue):")

if st.button("Generate Edit") and user_input:
    with st.spinner("🔍 Processing your request..."):
        result = identify_intent_and_rephrase(user_input)
        
    intent = result.get("intent", "other")
    prompt = result.get("prompt", "")
    print(prompt, "##################")

    if intent == "other":
        st.warning("⚠️ Sorry, This type of edit is not yet supported.")
    elif intent == "color":
        st.info("🎨 Proceeding with color edit...")
        with st.spinner("Generating color-edited image..."):
            output = generate_image(prompt, original_image_path, suffix="color")
            if output:
                st.success("✅ Image edited successfully!")
                image_container.image(output, caption="Edited Frock", width=400)
    elif intent == "sleeve":
        st.info("🧵 Proceeding with sleeve edit...")
        with st.spinner("Generating and selecting the best image..."):
            prompts = [f"{prompt} (variation {i})" for i in range(1, 4)]
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(generate_image, p, original_image_path, suffix=f"sleeve_{i}") for i, p in enumerate(prompts, start=1)]
                results = [f.result() for f in as_completed(futures)]
            
            scored_images = []
            for img in results:
                if img:
                    similarity = compare_images(original_image_path, img)
                    print(similarity,"simi.............")
                    scored_images.append((similarity, img))
            
            if scored_images:
                best = max(scored_images, key=lambda x: x[0])
                st.success(f"✅ Image edited successfully!")
                image_container.image(best[1], caption="Edited Frock", width=400)
            else:
                st.error("❌ Failed to generate a suitable image.")
