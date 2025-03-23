import torch
import torchvision
import torchvision.transforms as transforms
from spikingjelly.clock_driven import neuron, functional, encoding, surrogate
import torch.nn as nn

# Configurare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_window = 50

# Encoder Poisson
class PoissonEncoderWrapper(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.encoder = encoding.PoissonEncoder()
        self.time_steps = time_steps

    def forward(self, x):
        encoded = [self.encoder(x) for _ in range(self.time_steps)]
        return torch.stack(encoded)

# Arhitectura SNN
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)  # Creștem de la 100 la 256 neuroni
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

# Dataset și DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inițializare model
model = SimpleSNN().to(device)
encoder = PoissonEncoderWrapper(time_window).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Adăugat weight_decay
loss_fn = nn.CrossEntropyLoss()

# Evaluare
def evaluate(model, encoder, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            encoded = encoder(data)
            out = model(encoded)
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            functional.reset_net(model)
    accuracy = 100. * correct / total
    print(f"📊 Acuratețe: {accuracy:.2f}%")
    model.train()
    return accuracy

# Antrenare
n_epochs = 5
for epoch in range(n_epochs):
    print(f"🌟 Epoca {epoch + 1}/{n_epochs}")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        encoded = encoder(data)
        out = model(encoded)
        loss = loss_fn(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

        if batch_idx % 100 == 0:
            print(f"🔁 Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    evaluate(model, encoder, test_loader, device)

# Salvare
torch.save(model.state_dict(), 'snn_model.pth')
torch.save(encoder.state_dict(), 'snn_encoder.pth')
print("Model salvat!")