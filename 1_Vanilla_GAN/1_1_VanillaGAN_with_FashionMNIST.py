from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as moodels
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch, os

# MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"{device} is available.")

# Directory
model_dir = os.path.join('result/1/model')
os.makedirs(model_dir, exist_ok=True)

image_dir = os.path.join('result/1/sample')
os.makedirs(image_dir, exist_ok=True)

loss_dir = os.path.join('result/1/loss')
os.makedirs(loss_dir, exist_ok=True)

# Datasets
training_data = datasets.FashionMNIST(
    root='data/target/1',
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.FashionMNIST(
    root='data/target/1',
    train=False,
    download=False,
    transform=ToTensor()
)

# Hyperparameters
NOISE = 10
INPUT_SIZE = 28 * 28
BATCH_SIZE = 64
EPOCHS = 100

# Generator
class Generator(nn.Module):
    def __init__(self, noise):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(noise, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)  # reshape to (batch_size, 1, 28, 28)
        return img
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

# Generate Model
generator = Generator(NOISE).to(device)
discriminator = Discriminator(INPUT_SIZE).to(device)

# Loss, Optimizer
criterion = nn.BCELoss()
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4)
optimizer_generator = optim.Adam(generator.parameters(), lr=1e-4)

for param in discriminator.parameters():
    param.requires_grad = False

gan_input = torch.randn(BATCH_SIZE, NOISE).to(device)
x = generator(gan_input)
output = discriminator(x)

train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

# Visualize
def visualize_training(epoch, d_losses, g_losses):
    print(f'epoch: {epoch}, Discriminator Loss: {np.asarray(d_losses).mean():.4f}, Generator Loss: {np.asarray(g_losses).mean():.4f}')
    
# Save
def save_loss(epoch, d_losses, g_losses, loss_dir):
    os.makedirs(loss_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generatror Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'epoch: {epoch}, Discriminator Loss: {np.asarray(d_losses).mean():.4f}, Generator Loss: {np.asarray(g_losses).mean():.4f}')
    plt.savefig(os.path.join(loss_dir, f'generated_images_epoch_{epoch}.png'))
    plt.close()    

def save_sample(epoch, image_dir, NOISE):
    noise = torch.randn(24, NOISE).to(device)

    generator.eval()
    with torch.no_grad():
        generated_images = generator(noise).cpu().detach().numpy()

    generated_images = generated_images.reshape(-1, 28, 28) * 255
    generated_images = generated_images.astype(np.uint8)
    
    plt.figure(figsize=(8, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 6, i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f'generated_images_epoch_{epoch}.png'))
    plt.close()

# Train
d_losses = []
g_losses = []

for epoch in range(1, EPOCHS + 1):
    for i, (real_images, _) in enumerate(train_loader):

        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(device)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        for param in discriminator.parameters():
            param.requires_grad = True

        optimizer_discriminator.zero_grad()

        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        z = torch.randn(batch_size, NOISE).to(device)

        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()

        optimizer_discriminator.step()

        # Train Generator
        for param in discriminator.parameters():
            param.requires_grad = False
        
        optimizer_generator.zero_grad()

        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()

        optimizer_generator.step()
    
    d_losses.append(d_loss_real.item() + d_loss_fake.item())
    g_losses.append(g_loss.item())

    if epoch == 1 or epoch % 10 == 0:
        visualize_training(epoch, d_losses, g_losses)
        save_loss(epoch, d_losses, g_losses, loss_dir)
        save_sample(epoch, image_dir, NOISE)
        torch.save(generator.state_dict(), os.path.join(model_dir, f"generator.pth"))
        torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator.pth"))