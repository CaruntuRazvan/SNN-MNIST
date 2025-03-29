# app.py
import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from spikingjelly.clock_driven import neuron, functional, encoding, surrogate
import torch.nn as nn
import random

# 🔧 Configurare
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_window = 50

# 🔁 Encoder Poisson
class PoissonEncoderWrapper(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.encoder = encoding.PoissonEncoder()
        self.time_steps = time_steps

    def forward(self, x):
        encoded = [self.encoder(x) for _ in range(self.time_steps)]
        return torch.stack(encoded)

class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        self.fc2 = nn.Linear(256, 10)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

    def forward(self, x):
        mem = 0
        for t in range(x.shape[0]):
            out = self.flatten(x[t])
            out = self.fc1(out)
            out = self.sn1(out)
            out = self.fc2(out)
            out = self.sn2(out)
            mem += out
        return mem / x.shape[0]

# 🔄 Transformare și date de test
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

def get_random_samples(n=10):
    indices = random.sample(range(len(test_dataset)), n)
    images, labels = zip(*[test_dataset[i] for i in indices])
    return torch.stack(images), torch.tensor(labels)

# 🧠 Încarcă modelul și encoderul
model = SimpleSNN().to(device)
model.load_state_dict(torch.load('snn_model.pth', map_location=device))
model.eval()

encoder = PoissonEncoderWrapper(time_window).to(device)
encoder.load_state_dict(torch.load('snn_encoder.pth', map_location=device))
encoder.eval()

# 🎨 Interfață Streamlit
st.title("🧠 Clasificare MNIST cu SNN")
st.write("Acest demo folosește o rețea neuronală spiking (SNN) antrenată pe MNIST pentru a clasifica cifre.")

# 🔘 Buton 1: testare pe 10 imagini aleatoare
if st.button("🔁 Testează 10 imagini aleatoare"):
    images, labels = get_random_samples(n=10)
    images = images.to(device)
    labels = labels.to(device)

    encoded = encoder(images)
    with torch.no_grad():
        outputs = model(encoded)
        predictions = torch.argmax(outputs, dim=1)

    acc = (predictions == labels).sum().item() / len(labels)
    st.success(f"🎯 Acuratețea pe acest batch este: **{acc * 100:.2f}%**")

    # 🔢 Afișează imaginile
    st.subheader("🔍 Imagini testate:")
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()
    for i in range(10):
        img = images[i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Pred: {predictions[i].item()}\nLabel: {labels[i].item()}")
        axes[i].axis("off")
    st.pyplot(fig)

# 🔘 Buton 2: testare completă pe 1000 de imagini aleatorii
if st.button("✅ Evaluează pe 100 de imagini aleatorii"):
    st.info("Se evaluează modelul pe 100 de imagini din setul de test...")

    correct = 0
    total = 0
    incorrect_images = []  # Listă pentru a stoca imaginile greșit clasificate
    incorrect_labels = []  # Etichetele corespunzătoare pentru imaginile greșite
    incorrect_preds = []  # Predicțiile pentru imaginile greșite

    # Selectează aleatoriu 1000 de indici din setul de test
    random_indices = random.sample(range(len(test_dataset)), 100)

    with torch.no_grad():
        for idx in random_indices:
            img, label = test_dataset[idx]
            data = img.unsqueeze(0).to(device)
            encoded = encoder(data)
            output = model(encoded)
            pred = output.argmax().item()

            if pred == label:
                correct += 1
            else:
                # Adaugă imaginea și informațiile corespunzătoare pentru greșelile de predicție
                incorrect_images.append(img)
                incorrect_labels.append(label)
                incorrect_preds.append(pred)

            total += 1
            functional.reset_net(model)

    acc = correct / total * 100
    st.success(f"📊 Acuratețea pe 100 de imagini: **{acc:.2f}%**")

    # Afișează imaginile greșite
    # Afișează imaginile greșite
    if incorrect_images:
        st.subheader("🔴 Imagini greșit clasificate:")

        num_cols = 5  # Număr de imagini pe rând
        rows = [incorrect_images[i:i + num_cols] for i in range(0, len(incorrect_images), num_cols)]

        for row_idx, row in enumerate(rows):
            cols = st.columns(num_cols)
            for col_idx, col in enumerate(cols):
                img_idx = row_idx * num_cols + col_idx
                if img_idx < len(incorrect_images):
                    img = incorrect_images[img_idx]
                    pred = incorrect_preds[img_idx]
                    label = incorrect_labels[img_idx]

                    with col:
                        img_np = img.cpu().squeeze().numpy()
                        img_np = np.clip(img_np, 0, 1)

                        st.image(
                            img_np,
                            caption=f"Pred: {pred} | Label: {label}",
                            width=100,  # Dimensiune personalizată
                            use_container_width=False  # Parametrul corect
                        )
    else:
        st.info("Nu au fost greșeli de clasificare!")
