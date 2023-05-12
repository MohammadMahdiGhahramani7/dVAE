# !pip install torchinfo umap-learn

import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import umap
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings("ignore")

class Quantizer(nn.Module):
    def __init__(self, num_embs, emb_dim, Beta):

        super().__init__()
   
        self.num_embs = num_embs #vocab size (K in paper)
        self.emb_dim = emb_dim #space to quantize (D in paper)
        self.Beta = Beta

        self.embedding = nn.Embedding(self.num_embs, self.emb_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embs, 1/self.num_embs)
        
    def forward(self, z_e_x):
        # z_e_x: [B, D, H, W]
        x = z_e_x.permute(0, 2, 3, 1).contiguous()
        # x: [B, H, W, D]
        x_flat = x.view(-1, self.emb_dim) # [B*H*W, D]
        # x_flat: [B*H*W, D]
        distances = (torch.sum(x_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(x_flat, self.embedding.weight.t()))
        # distances: [B*H*W, K]

        e_i_stars_idx = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(e_i_stars_idx.shape[0], self.num_embs, device=z_e_x.device)
        encodings.scatter_(1, e_i_stars_idx, 1) # One hot vectorization: [B*H*W, K]
        
        # encodings:[B*H*W, K] * emb.weight[K, D] => quantized[B*H*W, D] => view: [B, H, W, D]
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)
        
        # Loss
        loss_q = F.mse_loss(quantized, x.detach())
        loss_e = F.mse_loss(quantized.detach(), x)
        loss = loss_q + self.Beta * loss_e
        
        quantized = x + (quantized - x).detach() #copy the gradient from z_q_x to z_e_x
        z_q_x = quantized.permute(0, 3, 1, 2).contiguous()
        # z_q_x: [B, D, H, W] -> similar to z_q_x

        probs = torch.mean(encodings, dim=0) #prob of selecting specific e_i
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))) #higher perplexity is better, Max: K
        
        return loss, z_q_x, perplexity, encodings
        
class Residual(nn.Module):
    def __init__(self, in_ch, num_res_hiddens, num_hiddens):

        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, num_res_hiddens, 3, 1, 1, bias=False)   
        self.conv2 = nn.Conv2d(num_res_hiddens, num_hiddens, 3, 1, 1, bias=False)
    
    def forward(self, x):

        Res = self.conv1(F.relu(x))
        Res = self.conv2(F.relu(Res))

        return x + Res


class ResidualStack(nn.Module):
    '''
    It does not change the shape, since <padding = kernel_size // 2> has been applied.
    '''
    def __init__(self, in_ch, num_layers, num_res_hiddens, num_hiddens):

        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([Residual(in_ch, num_res_hiddens, num_hiddens) for _ in range(self.num_layers)])

    def forward(self, x):

        for i in range(self.num_layers):
            x = self.layers[i](x)

        return F.relu(x)
        
class Encoder(nn.Module):
    def __init__(self, in_ch, num_layers, num_res_hiddens, num_hiddens):

        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, num_hiddens//2, 4, 2, 1)

        self.conv2 = nn.Conv2d(num_hiddens//2, num_hiddens, 4, 2, 1)

        self.conv3 = nn.Conv2d(num_hiddens, num_hiddens, 3, 1, 1)

        self.RES = ResidualStack(num_hiddens, num_layers, num_res_hiddens, num_hiddens)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = self.conv3(x)
        
        return self.RES(x)
        
class Decoder(nn.Module):
    def __init__(self, in_ch, num_layers, num_res_hiddens, num_hiddens):

        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, num_hiddens, 3, 1, 1)
        
        self.RES = ResidualStack(num_hiddens, num_layers, num_res_hiddens, num_hiddens)
        
        self.convTR1 = nn.ConvTranspose2d(num_hiddens, num_hiddens//2, 4, 2, 1)

        self.convTR2 = nn.ConvTranspose2d(num_hiddens//2, 3, 4, 2, 1)


    def forward(self, x):

        x = self.conv1(x)   
        x = self.RES(x)   
        x = F.relu(self.convTR1(x))
        
        return self.convTR2(x)
        
class dVAE(nn.Module):
    def __init__(self, num_res_layers, num_res_hiddens, num_hiddens, num_embs, emb_dim, Beta):
      
        super(Model, self).__init__()
        
        self.encoder = Encoder(3, num_res_layers, num_res_hiddens, num_hiddens)

        self.conv = nn.Conv2d(num_hiddens, emb_dim, 1, 1)

        self.QNTZ = Quantizer(num_embs, emb_dim, Beta)

        self.decoder = Decoder(emb_dim, num_res_layers, num_res_hiddens, num_hiddens)

    def forward(self, x):

        z_e_x = self.encoder(x)
        z_e_x = self.conv(z_e_x) #extra conv layer to set the number of channels to emb_dim=D

        loss, z_q_x, perplexity, _ = self.QNTZ(z_e_x)
        x_tilda = self.decoder(z_q_x)

        return loss, x_tilda, perplexity

def train(model, opt, dataloader, epochs, data_var, K):

  model.train()
  total_L, rec_L, vq_L, perplexities = [], [], [], []

  for epoch in range(epochs):
    losses, rec_losses, vq_losses, perps, step = 0, 0, 0, 0, 0
    for idx, (img, _) in enumerate(dataloader):

        img = img.to(device)
        optimizer.zero_grad()

        vq_loss, img_rec, perplexity = model(img)
        rec_loss= F.mse_loss(img_rec, img) / data_var
        loss = rec_loss + vq_loss
        
        vq_losses += vq_loss
        rec_losses += rec_loss
        losses += loss
        perps += perplexity

        loss.backward()
        optimizer.step()
        step += 1
    
    vq_L.append(vq_losses.item() / step)
    rec_L.append(rec_losses.item() / step)
    total_L.append(losses.item() / step)
    perplexities.append(perps.item() / step)

    if epoch % 1 == 0:
        print(f"Epoch: {epoch+1} -> Total_Loss: {total_L[-1]:.8f} ------ Rec_Loss: {rec_L[-1]:.8f} ------ VQ_Loss: {vq_L[-1]:.8f} ------ Perplexity: {perplexities[-1]:.2f} <= {K}")

  return total_L, rec_L, vq_L, perplexities

@torch.no_grad()
def generation(model, dataloader):

  (orgs, _) = next(iter(dataloader))
  orgs = orgs.to(device)
  recons = model.forward(orgs)[1]
  
  return (orgs.to('cpu'), recons.to('cpu')) 
  
def show(img):

    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
                              
# Set hyp-params
batch_size = 256
epochs = 2
lr = 1e-3
num_hiddens = 128
num_res_hiddens = 32
num_res_layers = 2
emb_dim = 64
num_embs = 512
Beta = 6.6

# Load the data
training_data = CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

data_var = np.var(training_data.data / 255.0)

validation_data = CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(validation_data, batch_size=32, shuffle=True, pin_memory=True)
                                 
# Model
model = dVAE(num_res_layers, num_res_hiddens, num_hiddens, num_embs, emb_dim, Beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)

# Training
start_time = time.time()
total_L, rec_L, vq_L, perplexities = train(model, optimizer, dataloader, epochs, data_var, num_embs)
end_time = time.time()
print(f"Training time: {end_time- start_time:.3f} seconds")

# Generate images
X, X_tilda = generation(model, val_dataloader)
lighter = 0.5

show(make_grid(X) + lighter)
show(make_grid(X_tilda) + lighter)

# Review Embeddings
proj = umap.UMAP(n_neighbors=3, min_dist=0.1, metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())
plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
plt.show()
