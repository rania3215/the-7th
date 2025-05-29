import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet
from PIL import Image
import gdown
import os
import pandas as pd

class EnsembleNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EnsembleNet, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()

        self.resnet101 = models.resnet101(pretrained=True)
        self.resnet101.fc = nn.Identity()

        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 2048 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x1 = self.resnet50(x)
        x2 = self.resnet101(x)
        x3 = self.efficientnet(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return self.classifier(x)

# -------------------------------
MODEL_ID = "1-x445n-cKYNiWLPWsYaLnHKCyX465NAj"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "7_emotions_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    model = EnsembleNet(num_classes=7)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_comments = {
    'Angry': "You look angry. Try to calm down by taking deep breaths: inhale slowly through your nose and exhale through your mouth.",
    'Disgust': "Disgust is a natural human response that protects you from harm — it’s okay to feel this way.",
    'Fear': "Rest and reassurance are important when feeling fear. You're not alone.",
    'Happy': "I'm so glad to see you happy — you're inspiring!",
    'Sad': "I'm sorry you're feeling this way. Remember, beautiful moments often follow deep sorrow.",
    'Surprise': "Life is full of unexpected and beautiful surprises!",
    'Neutral': "Okay! You're feeling neutral. Stay balanced!"
}

def preprocess_image(img):
    transform = transforms.Compose([
   transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
        transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

   




    img = img.convert("RGB")
    img = transform(img).unsqueeze(0)
    return img

# -------------------------------
st.title("The 7th Emotions")
st.write("**Done by:** Rania Otoum & Ghazal dabbas")
st.write("Upload an image and get your emotion prediction:")
# ----------------
import streamlit as st
import base64
logo = Image.open("logo.png")

# Inject CSS for floating image
st.markdown("""
    <style>
    .logo-img {
        position: fixed;
        top: 60px;
        right: 40px;
        width: 140px;
        height: 140px;
        border-radius: 50%;
        object-fit: cover;
        
        box-shadow: 0 0 6px rgba(0,0,0,0.2);
        z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

# Encode logo to Base64
def get_base64_image(image):
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

img_base64 = get_base64_image(logo)

# Add logo image as floating circular image
st.markdown(f"""
    <img src="data:image/png;base64,{img_base64}" class="logo-img">
""", unsafe_allow_html=True)
#=============


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Image", use_container_width=True)

    img_tensor = preprocess_image(image)

    with st.spinner("Analyzing emotions..."):
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            emotion = classes[predicted_idx]
            comment = emotion_comments[emotion]

    st.markdown(f"### Your Predicted Emotion: **{emotion}**")
    st.info(comment)

    st.subheader("Emotion Probabilities:")
    prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    st.bar_chart(pd.DataFrame(prob_dict, index=[0]))

