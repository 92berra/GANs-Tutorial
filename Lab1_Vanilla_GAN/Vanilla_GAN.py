from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn, torch.optim as optim, numpy as np, torch, argparse, os


# MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"{device} is available.")


# Default path
DEFAULT_TRAINING_DATA = './datasets'
DEFAULT_MODEL_DIR = './result/Lab1/model'
DEFAULT_IMAGES_DIR = './result/Lab1/images'
DEFAULT_LOSS_DIR = './result/Lab1/loss'


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
    

# Visualize
def visualize_training(epoch, d_losses, g_losses):
    print(f'epoch: {epoch}, Discriminator Loss: {np.asarray(d_losses).mean():.4f}, Generator Loss: {np.asarray(g_losses).mean():.4f}')
    

# Hyperparameters
DEFAULT_NOISE = 10
DEFAULT_INPUT_SIZE = 28 * 28
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100


# Train
def train(training_data, noise, input_size, batch_size, epochs, model_dir, images_dir, loss_dir):

    # Get datasets
    training_data = datasets.FashionMNIST(
        root='./datasets/',
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)


    # Generate save path
    model_dir = os.path.join(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    image_dir = os.path.join(images_dir)
    os.makedirs(image_dir, exist_ok=True)

    loss_dir = os.path.join(loss_dir)
    os.makedirs(loss_dir, exist_ok=True)


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


    # Generate Model
    generator = Generator(noise).to(device)
    discriminator = Discriminator(input_size).to(device)


    # Loss, Optimizer
    criterion = nn.BCELoss()
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer_generator = optim.Adam(generator.parameters(), lr=1e-4)

    for param in discriminator.parameters():
        param.requires_grad = False

    gan_input = torch.randn(batch_size, noise).to(device)
    x = generator(gan_input)
    output = discriminator(x)


    # Train
    d_losses = []
    g_losses = []

    for epoch in range(1, epochs + 1):
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

            z = torch.randn(batch_size, noise).to(device)

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
            save_sample(epoch, image_dir, noise)
            torch.save(generator.state_dict(), os.path.join(model_dir, f"generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator.pth"))


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_data', type=str, dest='training_data', default=DEFAULT_TRAINING_DATA, help='Path of datasets')
    parser.add_argument('--noise', type=int, dest='noise', default=DEFAULT_NOISE, help='Define noise value')
    parser.add_argument('--input_size', type=int, dest='input_size', default=DEFAULT_INPUT_SIZE, help='Define input size eg. 28x28, 256x256, etc.')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=DEFAULT_BATCH_SIZE, help='Define bacth size eg. 32, 64, 128, etc.')
    parser.add_argument('--epochs', type=int, dest='epochs', default=DEFAULT_EPOCHS, help='Define training epochs value')

    parser.add_argument('--model_dir', type=str, dest='model_dir', default=DEFAULT_MODEL_DIR, help='Directory to save checkpoint')
    parser.add_argument('--images_dir', type=str, dest='images_dir', default=DEFAULT_IMAGES_DIR, help='Directory to save generated images')
    parser.add_argument('--loss_dir', type=str, dest='loss_dir', default=DEFAULT_LOSS_DIR, help='Directory to save loss output')

    args = parser.parse_args()

    train(args.training_data, args.noise, args.input_size, args.batch_size, args.epochs, args.model_dir, args.images_dir, args.loss_dir)