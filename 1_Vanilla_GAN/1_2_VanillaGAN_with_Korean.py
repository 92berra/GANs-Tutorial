from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision.transforms import transforms
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch, os, glob, io

# MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"{device} is available.")

# Directory
model_dir = os.path.join('result/2/model')
os.makedirs(model_dir, exist_ok=True)

image_dir = os.path.join('result/2/sample')
os.makedirs(image_dir, exist_ok=True)

loss_dir = os.path.join('result/2/loss')
os.makedirs(loss_dir, exist_ok=True)

# Generate Font Image
TXT_FILE = 'data/characters/50characters.txt'
FONTS_DIR = 'data/fonts'
IMAGE_DIR = 'data/target/2'

if not os.path.exists(IMAGE_DIR):
    os.makedirs(os.path.join(IMAGE_DIR))

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

list_labels = []
with open(TXT_FILE, 'r', encoding='utf-8') as fr:
    for line in fr:
        list_labels.append(line.strip())

with io.open(TXT_FILE, 'r', encoding='utf-8') as f:
    labels = f.read().splitlines()

# Get a list of the fonts.
fonts = sorted(glob.glob(os.path.join(FONTS_DIR, '*.ttf')))
for f in fonts:
    filename = os.path.basename(f)
    filename_without_extension = os.path.splitext(filename)[0]
    print(filename_without_extension)

# Initialize numbers
total_count = 0
prev_count = 0
font_count = 1
char_no = 0
    
# Total number of font files is 
print('total number of fonts are ', len(fonts))

for character in labels:
    char_no += 1
        
    for font in fonts:
        total_count += 1

        image = Image.new('RGB', (IMAGE_WIDTH,IMAGE_HEIGHT), (0, 0, 0))
        w, h = image.size
                
        drawing = ImageDraw.Draw(image)
        font = ImageFont.truetype(font, 15)

        box = None
        new_box = drawing.textbbox((0, 0), character, font)
                
        new_w = new_box[2] - new_box[0]
        new_h = new_box[3] - new_box[1]
                
        box = new_box
        w = new_w
        h = new_h
                
        x = (IMAGE_WIDTH - w)//2 - box[0]
        y = (IMAGE_HEIGHT - h)//2 - box[1]

        drawing.text((x,y), character, fill=(255,255,255), font=font) 
        file_string = f'{char_no}.png'
        file_path = os.path.join(IMAGE_DIR, file_string)
        image.save(file_path, 'PNG')
        font_count += 1

    font_count = 1
char_no = 0
            
print('Finished generating {} images.'.format(total_count))
# Generate Font Image End

# Hyperparameters
NOISE = 10
INPUT_SIZE = 28*28
BATCH_SIZE = 32
EPOCHS = 100

# Datasets
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        image = image.view(-1)
        return image

# Transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# DataLoader
dataset = CustomDataset(root_dir='data/target/2', transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
        img = img.view(img.size(0), 1, 28, 28)
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
    param.requires_grad = True

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
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    noise = torch.randn(24, NOISE).to(device)

    generator.eval()
    with torch.no_grad():
        generated_images = generator(noise).cpu().numpy()
    
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
    for i, real_images in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(device)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        discriminator.train()
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
        generator.train()
        optimizer_generator.zero_grad()

        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()

        optimizer_generator.step()

        # Record losses
        d_losses.append(d_loss_real.item() + d_loss_fake.item())
        g_losses.append(g_loss.item())

    if epoch == 1 or epoch % 10 == 0:
        visualize_training(epoch, d_losses, g_losses)
        save_loss(epoch, d_losses, g_losses, loss_dir)
        save_sample(epoch, image_dir, NOISE)
        torch.save(generator.state_dict(), os.path.join(model_dir, f"generator.pth"))
        torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator.pth"))