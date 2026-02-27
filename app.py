import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SASYA RAKSHAK | Mob-Res + SE",
    page_icon="🌿",
    layout="wide"
)

# -------------------- LOAD MODEL (.pkl contains Keras model) --------------------
@st.cache_resource
def load_model():
    with open("mob_res_se_final.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------- CLASS NAMES --------------------
CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy',
    'Corn___Cercospora_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight','Corn___healthy',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy',
    'Pepper___Bacterial_spot','Pepper___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# -------------------- PREPROCESS --------------------
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------- GRAD-CAM --------------------
def get_gradcam_heatmap(model, img_array, layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)

        # Handle pickle edge-case
        if isinstance(preds, list):
            preds = preds[0]

        pred_class = tf.argmax(preds[0])
        class_score = preds[:, pred_class]

    grads = tape.gradient(class_score, conv_out)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap * 0.4 + original_img
    return np.uint8(overlay)

# -------------------- UI --------------------
st.title("🌿 SASYA RAKSHAK")
st.subheader("Mob-Res + SE Attention | Edge-AI Crop Health Assessment")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    processed = preprocess_image(image)

    preds = model.predict(processed)[0]

    # ---- Top 3 ----
    top3_idx = np.argsort(preds)[::-1][:3]
    top3_classes = [CLASS_NAMES[i] for i in top3_idx]
    top3_probs = [preds[i] * 100 for i in top3_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Leaf", width=450)

    with col2:
        st.markdown("### 🔎 Predicted Class")
        st.success(top3_classes[0])

        st.markdown("### 📊 Confidence")
        st.progress(int(top3_probs[0]))
        st.write(f"{top3_probs[0]:.2f}%")

        st.markdown("### 📈 Top-3 Probabilities")
        for i in range(3):
            st.write(f"{i+1}. {top3_classes[i]} — {top3_probs[i]:.2f}%")
            st.progress(int(top3_probs[i]))

    # -------------------- GRAD-CAM SECTION --------------------
    st.markdown("---")
    st.markdown("## 🔍 Explainable AI (Grad-CAM)")

    try:
        heatmap = get_gradcam_heatmap(model, processed, "conv2d_8")
        overlay = overlay_heatmap(img_array, heatmap)

        # Centered image block
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            st.image(overlay, caption="Grad-CAM (Residual Path)", width=500)

        st.info("Red regions indicate strong diagnostic activation.")

    except Exception as e:
        st.error("Grad-CAM could not be generated.")
        st.write(str(e))

st.markdown("---")
st.caption("Hybrid Lightweight CNN | SE Attention Fusion | Explainable AI Enabled")