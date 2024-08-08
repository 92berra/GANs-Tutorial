import torch
import torch.utils.data
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import os, argparse
import matplotlib.pyplot as plt



# MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"{device} is available.")


# Default path
DEFAULT_DATASET_DIR = './datasets/celeba'
DEFAULT_MODEL_DIR = './result/Lab2/model'
DEFAULT_IMAGES_DIR = './result/Lab2/images'
DEFAULT_LOSS_DIR = './result/Lab2/loss'


# Hyperparameters
WORKERS = 2 # Number of thread for dataloader
BATCH_SIZE = 128 # Batch size
IMAGE_SIZE = 64 # Image size
NC = 3 # Number of Channel, RGB is 3
NZ = 100 # Size of latency space(input of generator)
NGF = 64 # Size of feature data pass through G
NDF = 64 # Size of feature data pass through D
EPOCHS = 5 # Number of epoch
LEARNING_RATE = 0.0002 # Learning rate of optimizer 
BETA1 = 0.5 # beta1 Hypterparameter of Adam optimizer
NGPU = 1 # Number of GPU. If cpu, text 0.


# Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
            nn.ConvTranspose2d( NZ, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(NGF*4) x 8 x 8``
            nn.ConvTranspose2d( NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기. ``(nc) x 64 x 64``
        )
    def forward(self, input):
        return self.main(input)
    
    
# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            
            # 입력 데이터의 크기는 ``(nc) x 64 x 64`` 입니다
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 위의 계층을 통과한 데이터의 크기. ``(NDF) x 32 x 32``
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 위의 계층을 통과한 데이터의 크기. ``(NDF*2) x 16 x 16``
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 위의 계층을 통과한 데이터의 크기. ``(NDF*4) x 8 x 8``
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 위의 계층을 통과한 데이터의 크기. ``(NDF*8) x 4 x 4``
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    

# Train
def train(dataset_dir, workers, batch_size, image_size, nc, nz, ngf, ndf, epochs, learning_rate, beta1, ngpu, model_dir, images_dir, loss_dir):
    
    
    # Get datasets
    training_data = dset.ImageFolder(
        root=dataset_dir,
        transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    train_loader = torch.utils.data.DataLoader(training_data, 
                                               batch_size=batch_size,
                                               shuffle=True, 
                                               num_workers=workers)


    # Generate save path
    model_dir = os.path.join(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    image_dir = os.path.join(images_dir)
    os.makedirs(image_dir, exist_ok=True)

    loss_dir = os.path.join(loss_dir)
    os.makedirs(loss_dir, exist_ok=True)
    
    # Customize weight initialization function (apply to netG, netD)
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    
    # Create Generator
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init) # Initialize (mean=0, stdev=0.02)
    print(f'Generator: {netG}') # Check Generator Architecture
    
    
    # Create Discriminator
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init) # Initialize (mean=0, stdev=0.02)
    print(netD) # Check Generator Architecture
    
    
    # Save
    def save_loss(epoch, D_losses, g_losses, loss_dir):
        os.makedirs(loss_dir, exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.plot(D_losses, label='Discriminator Loss')
        plt.plot(G_losses, label='Generatror Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'epoch: {epoch}, Discriminator Loss: {np.asarray(D_losses).mean():.4f}, Generator Loss: {np.asarray(G_losses).mean():.4f}')
        plt.savefig(os.path.join(loss_dir, f'generated_images_epoch_{epoch}.png'))
        plt.close()    

    def save_sample(epoch, image_dir, NZ):
        noise = torch.randn(24, NZ, 1, 1).to(device)  # Reshape noise to (24, 100, 1, 1)
        netG.eval()
        with torch.no_grad():
            generated_images = netG(noise).cpu()
        generated_images = (generated_images + 1) / 2.0  # Denormalize the images

        plt.figure(figsize=(8, 4))
        for i in range(generated_images.size(0)):
            plt.subplot(4, 6, i + 1)
            plt.imshow(np.transpose(vutils.make_grid(generated_images[i], padding=2, normalize=True), (1, 2, 0)))
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f'generated_images_epoch_{epoch}.png'))
        plt.close()

        
        
    # Initialize BCELoss
    criterion = nn.BCELoss()

    # Create latency space vector to check G 
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Fix True/False label for training
    real_label = 1.
    fake_label = 0.

    # Create Adam Optimizer for G, D
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Save loss value for training
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    
    # Iterate
    for epoch in range(EPOCHS):
        
        # Batch iteration per epoch
        for i, data in enumerate(train_loader, 0):

            # Update the value of netD
            # Maximize the value : log(D(x)) + log(1 - D(G(z)))
            # Train real data
            netD.zero_grad()
            
            # Modify below values(device, batch_size, label)
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 
                               real_label,
                               dtype=torch.float, 
                               device=device)
            
            # Pass real data to discriminator
            output = netD(real_cpu).view(-1)
            
            # Calculate loss value
            errD_real = criterion(output, label)
            
            # Calculate mean value
            errD_real.backward()
            D_x = output.mean().item()

            # Train fake data
            # Create latency space vector to netG
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            
            # Create fake image using netG
            fake = netG(noise)
            label.fill_(fake_label)
            
            # Discriminator : real or fake
            output = netD(fake.detach()).view(-1)
            
            # Calculate loss value of netD
            errD_fake = criterion(output, label)
            
            # Calculate mean value accumulating prior mean value
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # fake loss + real loss
            # errD will not use during backward, but report training condition
            errD = errD_real + errD_fake
            
            # Update netD
            optimizerD.step()


            # Update netG
            # Maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  # We will use this label to calculate G loss.
            
            
            # Repass fake data to D because updated D
            # G was not updated, D was updated. Differ from prior loss value
            output = netD(fake).view(-1)
            
            
            # Calculate G loss
            errG = criterion(output, label)
            
            
            # Calculate G mean
            errG.backward()
            D_G_z2 = output.mean().item()
            
            
            # Update G
            optimizerG.step()
            
            
            # Save loss value for graph
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            
            # Visualize
            if i % 50 == 0:
                print(f'[{epoch}/{EPOCHS}] [{i}/{len(train_loader)}] Loss_D: {errD.item()}, Loss_G: {errG.item()}, D(x): {D_x}, D(G(z)): {D_G_z1} / {D_G_z2}')


            # Save output of G passed fixed_noise
            if (iters % 500 == 0) or ((epoch == EPOCHS-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                
            # Print training step
            if epoch == 1 or epoch % 10 == 0:
                save_loss(epoch, D_losses, G_losses, loss_dir)
                save_sample(epoch, image_dir, NZ)
                torch.save(netG.state_dict(), os.path.join(model_dir, f"generator.pth"))
                torch.save(netD.state_dict(), os.path.join(model_dir, f"discriminator.pth"))

            iters += 1
            
            
# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_dir', type=str, dest='dataset_dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--workers', type=str, dest='workers', default=WORKERS)
    parser.add_argument('--batch_size', type=str, dest='batch_size', default=BATCH_SIZE)
    parser.add_argument('--image_size', type=str, dest='image_size', default=IMAGE_SIZE)
    parser.add_argument('--nc', type=str, dest='nc', default=NC)
    parser.add_argument('--nz', type=str, dest='nz', default=NZ)
    parser.add_argument('--ngf', type=str, dest='ngf', default=NGF)
    parser.add_argument('--ndf', type=str, dest='ndf', default=NDF)
    parser.add_argument('--epochs', type=str, dest='epochs', default=EPOCHS)
    parser.add_argument('--learning_rate', type=str, dest='learning_rate', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=str, dest='beta1', default=BETA1)
    parser.add_argument('--ngpu', type=str, dest='ngpu', default=NGPU)
    parser.add_argument('--model_dir', type=str, dest='model_dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--images_dir', type=str, dest='images_dir', default=DEFAULT_IMAGES_DIR)
    parser.add_argument('--loss_dir', type=str, dest='loss_dir', default=DEFAULT_LOSS_DIR)

    args = parser.parse_args()
    
    train(args.dataset_dir, args.workers, args.batch_size, args.image_size, args.nc, args.nz, args.ngf, args.ndf, args.epochs, args.learning_rate, args.beta1, args.ngpu, args.model_dir, args.images_dir, args.loss_dir)