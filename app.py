import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import SteelCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches"
]

@st.cache_resource
def load_model():
    model = SteelCNN().to(device)

    model.load_state_dict(
        torch.load("steel_defect_cnn.pth", map_location=device)
    )

    model.eval()

    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

st.set_page_config(
    page_title="Steel Defect Classifier",
    layout="centered"
)

st.title("Steel Surface Defect Classification")

st.write(
    "Upload a steel surface image to identify its defect type."
)

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["bmp", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded Steel Surface Image",
        width=300
    )
    if st.button("Classify Image"):

        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = F.softmax(outputs, dim=1)

            confidence, predicted = torch.max(
                probabilities, dim=1
            )
        predicted_class = class_names[predicted.item()]

        confidence_score = confidence.item() * 100

        st.success("Classification Completed!")

        st.subheader("Prediction")

        st.write(f"**Defect Type:** {predicted_class}")

        st.write(f"**Confidence:** {confidence_score:.2f}%")

        st.subheader("Class Probabilities")

        for i, class_name in enumerate(class_names):

            probability = probabilities[0][i].item() * 100

            st.write(f"{class_name}: {probability:.2f}%")