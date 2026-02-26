import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SASYA RAKSHAK | Edge-AI Crop Diagnostics",
    page_icon="🌿",
    layout="wide"
)

# -------------------- CUSTOM CSS THEME --------------------
st.markdown("""
<style>

body {
    background-color: #0f1f16;
}

.main {
    background: linear-gradient(135deg, #0f1f16, #1e3a2b);
    color: #e6f2e9;
}

h1, h2, h3 {
    color: #d4f5dc;
    font-weight: 600;
}

.section-card {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

.metric-box {
    background-color: rgba(34, 139, 34, 0.2);
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
}

.status-healthy {
    color: #4CAF50;
    font-weight: bold;
}

.status-disease {
    color: #ff6b6b;
    font-weight: bold;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    with open('mob_res_se_final.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------- CLASS LABELS --------------------
CLASS_NAMES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 
    'Cherry - Powdery Mildew', 'Cherry - Healthy', 
    'Corn - Cercospora Leaf Spot / Gray Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy', 
    'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf Blight (Isariopsis Leaf Spot)', 'Grape - Healthy', 
    'Orange - Haunglongbing (Citrus Greening)', 
    'Peach - Bacterial Spot', 'Peach - Healthy', 
    'Pepper (Bell) - Bacterial Spot', 'Pepper (Bell) - Healthy', 
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy', 
    'Raspberry - Healthy', 
    'Soybean - Healthy', 
    'Squash - Powdery Mildew', 
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy', 
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold', 
    'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot', 
    'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
]

# -------------------- PREPROCESS --------------------
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image)

    if len(img_array.shape) == 2 or img_array.shape[-1] != 3:
        image = image.convert("RGB")
        img_array = np.array(image)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -------------------- HEADER --------------------
st.markdown("""
# 🌿SASYA RAKSHAK  
### Attention-Enhanced Edge-AI for Real-Time Crop Health Assessment
""")

st.markdown("""
<div class="section-card">
<b>Architecture:</b> Mob-Res + SE (Dual Pathway Hybrid CNN)<br>
<b>Design Philosophy:</b> Lightweight | Robust | Explainable<br>
<b>Deployment Target:</b> Offline Mobile Edge-AI Systems
</div>
""", unsafe_allow_html=True)


# -------------------- MAIN LAYOUT --------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📤 Upload Leaf Image")
    uploaded_file = st.file_uploader("Supported formats: JPG / PNG", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Leaf Sample", use_container_width=True)


with col2:
    st.markdown("## 🧠 Diagnostic Output")

    if uploaded_file:
        with st.spinner("Executing Dual-Pathway Feature Extraction..."):

            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)

            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = predictions[0][predicted_class_index] * 100

            st.markdown(f"### 🔎 Predicted Class")
            st.markdown(f"<div class='metric-box'>{predicted_class_name}</div>", unsafe_allow_html=True)

            st.markdown("### 📊 Confidence Score")
            st.progress(int(confidence))
            st.write(f"{confidence:.2f}% certainty")

            if "Healthy" in predicted_class_name:
                st.markdown("<p class='status-healthy'>✔ Leaf Condition: Healthy</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='status-disease'>⚠ Disease Detected – Intervention Recommended</p>", unsafe_allow_html=True)


# -------------------- ARCHITECTURE SECTION --------------------
st.markdown("---")
st.markdown("## ⚙ System Architecture Overview")

st.markdown("""
<div class="section-card">

<b>Stage 1:</b> Image Normalization (128×128) <br>
<b>Stage 2:</b> Dual Feature Extraction  
&nbsp;&nbsp;&nbsp;&nbsp;• Residual Blocks → Fine-Grained Lesion Textures  
&nbsp;&nbsp;&nbsp;&nbsp;• MobileNetV2 → Global Structural Patterns  

<b>Stage 3:</b> Spatial SE Attention (Feature Prioritization)  
<b>Stage 4:</b> Feature Concatenation (1536-D Fusion Vector)  
<b>Stage 5:</b> Channel Attention Recalibration  
<b>Stage 6:</b> Softmax Classification (38 Classes)  

</div>
""", unsafe_allow_html=True)


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<center>
AI-Powered Precision Agriculture | Edge-Optimized Deep Learning | Explainable AI Integration  
</center>
""", unsafe_allow_html=True)