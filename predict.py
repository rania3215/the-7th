import torch
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# استيراد النموذج إذا كان معرف مسبقًا
from model import build_model
model = build_model()
model.load_state_dict(torch.load("/content/emotion_app/7_emotions_model.pth", map_location=device))

model.eval().to(device)

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_image(image_path):
    image = Image.open(image_path).convert("L").resize((224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]
