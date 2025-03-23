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
time_window = 20

# 🔁 Encoder Poisson
class PoissonEncoderWrapper(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.encoder = encoding.PoissonEncoder()
        self.time_steps = time_steps

    def forward(self, x):
        encoded = [self.encoder(x) for _ in range(self.time_steps)]
        return torch.stack(encoded)

# 🧠 Rețea SNN
# 🧠 Rețea SNN
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)  # Modificat de la 100 la 256
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        self.fc2 = nn.Linear(256, 10)     # Modificat de la 100 la 256
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

# 🔘 Buton 2: testare completă pe 100 de imagini
if st.button("✅ Evaluează pe 100 de imagini"):
    st.info("Se evaluează modelul pe 100 de imagini din setul de test...")

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(100):
            img, label = test_dataset[i]
            data = img.unsqueeze(0).to(device)
            encoded = encoder(data)
            output = model(encoded)
            pred = output.argmax().item()
            if pred == label:
                correct += 1
            total += 1
            functional.reset_net(model)

    acc = correct / total * 100
    st.success(f"📊 Acuratețea pe 100 de imagini: **{acc:.2f}%**")