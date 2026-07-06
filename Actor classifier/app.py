import streamlit as st
import torch
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from streamlit_cropper import st_cropper

import models
from config import config

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Tamil Actor Classifier",
    page_icon="🎭",
    layout="wide"
)

CLASSES = [
    "Ajith_Kumar",
    "Anushka_Shetty",
    "Dhanush",
    "Nayanthara",
    "Rajinikanth",
    "Samantha",
    "Simbu",
    "Sivakarthikeyan",
    "Suriya",
    "Tamannah",
    "Trisha",
    "Vijay",
    "Vijay_Sethupathi"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = models.convnext(
        "base",
        len(CLASSES)
    )

    model.load_state_dict(
        torch.load(
            "../models/ConvNextBase.pth",
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


MODEL = load_model()

# --------------------------------------------------
# FACE DETECTION
# --------------------------------------------------

def detect_face(image: Image.Image):

    img_cv = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        img_cv,
        cv2.COLOR_BGR2GRAY
    )

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        return image

    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    margin = int(0.30 * max(w, h))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)

    x2 = min(img_cv.shape[1], x + w + margin)
    y2 = min(img_cv.shape[0], y + h + margin)

    crop = img_cv[y1:y2, x1:x2]

    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(crop)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(image):

    x = config.TEST_TRANSFORM(image)

    x = x.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        logits = MODEL(x)

        probs = torch.softmax(
            logits,
            dim=1
        ).squeeze()

    df = pd.DataFrame({
        "Actor": CLASSES,
        "Probability (%)":
            probs.cpu().numpy() * 100
    })

    df = df.sort_values(
        "Probability (%)",
        ascending=False
    )

    return df

# --------------------------------------------------
# UI
# --------------------------------------------------

st.sidebar.title("About")

st.sidebar.markdown(
    """
    ### Tamil Actor Classifier

    Built using:
    - PyTorch
    - ConvNeXt Base
    - Streamlit

    **Model Accuracy:** 92.05%

    ### Links

    🔗 [GitHub Repository](https://github.com/monishwarmc/Actor-classifier)

    📊 [Kaggle Dataset](https://www.kaggle.com/datasets/monishwarmc/southindianactorsimages)
    """
)

st.title("📽️ Tamil Actor Classifier")

st.write(
    """
    Upload an image of a South Indian actor.\n
    You can also upload your Photo to know which actor you resemble the most.\n
    This Model is trained only with 'Ajith_Kumar', 'Anushka_Shetty', 'Dhanush', 'Nayanthara', 'Rajinikanth', 'Samantha', 'Simbu', 'Sivakarthikeyan', 'Suriya', 'Tamannah', 'Trisha', 'Vijay' and 'Vijay_Sethupathi'
    """
    )

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    auto_crop = detect_face(image)

    st.subheader("Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Original Image",
            width="stretch"
        )

    with col2:
        st.subheader("Auto cropped face")
        st.image(
            auto_crop,
            width="stretch",
            caption="The app will automatically detect a face.\nIf the crop is incorrect, adjust it manually before prediction for better result"
        )

    st.divider()


    use_manual_crop = st.checkbox(
        "Manually adjust face crop"
    )

    if use_manual_crop:
        st.subheader("Adjust Crop")
        final_crop = st_cropper(
            image,
            realtime_update=True,
            aspect_ratio=None,
            return_type="image"
        )
    else:
        final_crop = auto_crop

    st.image(
        final_crop,
        caption="Final Image Used For Prediction",
        width=350
    )

    if st.button(
        "Predict",
        type="primary"
    ):

        df = predict(final_crop)

        predicted_actor = df.iloc[0]["Actor"]
        confidence = df.iloc[0]["Probability (%)"]

        if confidence >= 70:
            st.success(
                f"🎭 Predicted Actor: {predicted_actor} ({confidence:.2f}% confidence)"
            )
        elif confidence >= 50:
            st.warning(
                f"🎭 Most Likely Actor: {predicted_actor} ({confidence:.2f}% confidence)"
            )
        else:
            st.warning(
                f"⚠️ Closest Match: {predicted_actor} ({confidence:.2f}% confidence)"
            )
        
        st.info(
            """
            💡 If the prediction seems incorrect,\nadjust the crop so that the face is clearly visible and fills most of the image.
            """
        )

        st.subheader("Top 3 Predictions")

        st.table(
            df.head(3)
            .reset_index(drop=True)
        )

        st.subheader("Confidence Distribution")

        chart_df = df.set_index("Actor")

        st.bar_chart(chart_df)

        st.subheader("All Probabilities")

        st.dataframe(
            df,
            width="stretch"
        )