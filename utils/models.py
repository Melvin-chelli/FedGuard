from typing import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os

# Hard coding the value for testing purpose
flat_shape = [784]
cond_shape=10
intermediate_dim=400
z_dim=20

PRINT_REQ = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class Encoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_z):

        super().__init__()
         # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=5, stride=1,padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2,padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1,padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2,padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=7, stride=1,padding='valid')
        self.lin1 = nn.Linear(in_features=80, out_features=20)
        self.lin2 = nn.Linear(in_features=80, out_features=20)

        # reparameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, inputs, device=DEVICE):
        x = inputs[0].to(device)
        y = inputs[1].to(device)

        # y = F.one_hot(y, 10).to(device)
        y = y.view(-1, 10, 1, 1).to(device)

        ones = torch.ones(x.size()[0], 
                                10,
                                x.size()[2], 
                                x.size()[3], 
                                dtype=x.dtype).to(device)
        y = ones * y
        x = torch.cat((x, y), dim=1)

        print_debug(f"input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        print_debug(x.shape)
        # 32, 28, 28
        x = F.pad(x, (0,3,0,3))
        print_debug(x.shape)
        # 32, 31, 31
        x = F.relu(self.conv2(x))
        print_debug(x.shape)
        # 32, 14, 14
        x = F.relu(self.conv3(x))
        print_debug(x.shape)
        # 64, 14, 14
        x = F.pad(x, (0,3,0,3))
        print_debug(x.shape)
        # 64, 17, 17
        x = F.relu(self.conv4(x))
        print_debug(x.shape)
        # 64, 7, 7
        x = F.relu(self.conv5(x))
        print_debug(x.shape)
        # 80, 1, 1
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        print_debug(f"After flatten shape: {x.shape}")
        # 80
        # print_debug(f"Concatenating {x.shape} with {y.shape}")
        # concat = torch.cat([x, y], dim=-1)
        # print_debug(f"After concatenation shape: {concat.shape}")
        # 90
        # loc=torch.zeros(mu_logvar.shape)
        # scale=torch.ones(mu_logvar.shape)
        # diagn = Independent(Normal(loc, scale), 1)
        mu = self.lin1(x)
        print_debug(f"mu shape: {mu.shape}")
        # 20
        logvar = self.lin2(x)
        print_debug(f"logvar shape: {logvar.shape}")
        z = self.reparameterize(mu, logvar)
        print_debug(f"Returning shape {z.shape}")
        return  mu, logvar, z



class Decoder(nn.Module):
    def __init__(self, dim_y, dim_z):
        super().__init__()
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.deconv1 = nn.ConvTranspose2d(in_channels=30, out_channels=64, kernel_size=7, stride=1, padding=0) # valid means no pad
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1) # pad operation added in forward
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1) # pad operation added in forward
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1,padding='same')
        

    def forward(self, inputs, device=DEVICE):
        x = inputs[0].to(device)#.unsqueeze(dim=0)
        y = inputs[1].to(device)
        print_debug(f"latent space shape: {x.shape}, labels shape: {y.shape}")
        x = torch.cat([x, y], dim=1)
        x = torch.reshape(x, (-1, self.dim_z+self.dim_y, 1, 1))
        print_debug(f"After concatenation shape: {x.shape}")
        x = F.relu(self.deconv1(x))
        print_debug(f"ConvTrans1 output shape: {x.shape}")
        x = F.relu(self.deconv2(x))
        print_debug(f"ConvTrans2 output shape: {x.shape}")
        x = F.pad(x, (0,0,0,0))
        x = F.relu(self.deconv3(x))
        print_debug(f"ConvTrans3 output shape: {x.shape}")
        x = F.relu(self.deconv4(x))
        print_debug(f"ConvTrans4 output shape: {x.shape}")
        # x = F.pad(x, (0,3,0,3))
        x = F.relu(self.deconv5(x))
        print_debug(f"ConvTrans5 output shape: {x.shape}")
        x = F.relu(self.deconv6(x))
        print_debug(f"ConvTrans6 output shape: {x.shape}")
        x = torch.sigmoid(self.conv(x))
        print_debug(f"Conv output shape: {x.shape}")
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        print_debug(f"After flatten shape: {x.shape}")
        return x



class CVAE_big(nn.Module):
    def __init__(self, dim_x, dim_y, dim_z):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flat_shape[0], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=dim_y),
            nn.Softmax(dim=-1)
        )

        #Encoder 
        self.encoder = Encoder(dim_x=dim_x, dim_y=dim_y, dim_z=dim_z)

        #Decoder
        self.decoder = Decoder(dim_y=dim_y, dim_z=dim_z)

    def forward(self, inputs, device=DEVICE):
        x, y = inputs      
        x = x.to(device)
        y = F.one_hot(y, 10).to(device)  
        print_debug(f"Inputs shape: {x.shape} and labels: {y.shape}")
        c_out = self.classifier(x)
        mu, logvar, z = self.encoder((x,y))
        out = self.decoder((z, y))
        print_debug(f"decoder output shape is: {out.shape}")
        return mu, logvar, out, c_out

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k.removeprefix("classifier.").removeprefix("decoder."): torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.classifier.load_state_dict(state_dict, strict=False)
        self.decoder.load_state_dict(state_dict, strict=False)

   
def print_debug(data):
    if PRINT_REQ:
        print(data)
    else:
        pass


class Classifier_small(nn.Module):
    def __init__(self, dim_y):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flat_shape[0], out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=dim_y),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs, device=DEVICE):
        x = inputs.to(device)
        c_out = self.classifier(x)
        return c_out

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


class Classifier(nn.Module):
    def __init__(self, dim_y):
        """
        McMahan et al., 2016; 1,663,370 parameters
        """
        super(Classifier, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32 * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(32 * 2) * (7 * 7), out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, device=DEVICE):
        x = inputs.to(device)
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


class VAE(nn.Module):
    def __init__(self,input_dim=784, latent_dim=20,hidden_dim=500):
        super(VAE,self).__init__()
        self.fc_e1 = nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, input_dim)
        
        self.input_dim = input_dim
    def encoder(self,x_in):
        x = F.relu(self.fc_e1(x_in.view(-1,self.input_dim)))
        x = F.relu(self.fc_e2(x))
        mean = self.fc_mean(x)
        logvar = F.softplus( self.fc_logvar(x) )
        return mean, logvar
    
    def decoder(self,z):
        z = F.relu(self.fc_d1(z))
        z = F.relu(self.fc_d2(z))
        x_out = F.sigmoid(self.fc_d3(z))
        return x_out.view(-1, self.input_dim)
    
    def sample_normal(self,mean,logvar):
        sd = torch.exp(logvar*0.5).to(DEVICE)
        e = Variable(torch.randn(sd.size())).to(DEVICE) # Sample from standard normal
        z = e.mul(sd).add_(mean).to(DEVICE)
        return z
    
    def forward(self,x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean,z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

    def test(self, input_data, device):
        running_loss = []
        for single_x in input_data:
            single_x = torch.tensor(single_x).float().to(device)

            x_in = Variable(single_x).to(device)
            x_out, z_mu, z_logvar = self.forward(x_in)
            # loss = self.criterion(x_out, x_in, z_mu, z_logvar)
            x_out = x_out.view(-1)
            x_in = x_in.view(-1)
            bce_loss = F.mse_loss(x_out, x_in, size_average=False)
            kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
            loss = (bce_loss + kld_loss)

            running_loss.append(loss.item())
        return running_loss


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, inputs, device=DEVICE):
        x = inputs.to(device)
        x = x.view(-1, 28*28)
        out = self.linear(x)
        return out

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


# class CVAE_regression(nn.Module):
#     def __init__(self, dim_x, dim_y, dim_z, input_size, num_classes):
#         super(CVAE_regression, self).__init__()
#         self.linear = nn.Linear(input_size, num_classes)

#         #Encoder 
#         self.encoder = Encoder(dim_x=dim_x, dim_y=dim_y, dim_z=dim_z)

#         #Decoder
#         self.decoder = Decoder(dim_y=dim_y, dim_z=dim_z)

#     def forward(self, inputs, device=DEVICE):
#         x, y = inputs      
#         x = x.to(device)
#         y = F.one_hot(y, 10).to(device)
#         c_out = self.linear(x.view(-1, 28*28))
#         print_debug(f"Inputs shape: {x.shape} and labels: {y.shape}")
#         mu, logvar, z = self.encoder((x,y))
#         out = self.decoder((z, y))
#         print_debug(f"decoder output shape is: {out.shape}")
#         return mu, logvar, out, c_out

#     def set_weights(self, weights):
#         """Set model weights from a list of NumPy ndarrays."""
#         state_dict = OrderedDict(
#             {k.removeprefix("linear.").removeprefix("decoder."): torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
#         )
#         self.linear.load_state_dict(state_dict, strict=False)
#         self.decoder.load_state_dict(state_dict, strict=False)


class DenseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_in = nn.Linear(in_features=z_dim+cond_shape, out_features=intermediate_dim)
        self.dec_out = nn.Linear(in_features=intermediate_dim, out_features=flat_shape[0]+cond_shape)

    def forward(self, inp):
        z, y = inp
        t = torch.cat((z,y), dim=1)

        dec_in_ret = F.relu(self.dec_in(t))
        dec_out = torch.sigmoid(self.dec_out(dec_in_ret))
        return dec_out


class DenseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_in = nn.Linear(in_features=flat_shape[0]+cond_shape, out_features=intermediate_dim)
        self.mu = nn.Linear(in_features=intermediate_dim, out_features=z_dim)
        self.logvar = nn.Linear(in_features=intermediate_dim, out_features=z_dim)

    def forward(self, inp):
        x = inp[0].to(DEVICE)
        y = inp[1].to(DEVICE)

        flat_data = x.view(-1, flat_shape[0]).to(DEVICE)     
        t = torch.cat((flat_data, y), 1)

        enc_in_ret = F.relu(self.enc_in(t))
        mu_out = F.relu(self.mu(enc_in_ret))
        logvar_out = F.relu(self.logvar(enc_in_ret))
        return mu_out, logvar_out

    def reparametarize(self, mu, logvar):
        sd = torch.exp(0.5*logvar)
        eps = torch.randn_like(sd)
        return mu + eps*sd


class CVAE(nn.Module):
    def __init__(self, dim_x, dim_y, dim_z, classifier="cnn"):
        super(CVAE, self).__init__()

        if classifier == "cnn":
            self.classifier = Classifier(dim_y=dim_y)
        elif classifier == "regression":
            self.classifier = LogisticRegression(input_size=28*28, num_classes=10)

        self.decoder = DenseDecoder()
        self.encoder = DenseEncoder()

    def forward(self, inp, train_cvae=True, train_classifier=True, device=DEVICE):
        x, y = inp
        x = x.to(device)

        if train_classifier:
            c_out = self.classifier(x)
        if train_cvae:
            y = F.one_hot(y, 10).to(device) 
            mu, logvar = self.encoder((x, y))
            z = self.encoder.reparametarize(mu, logvar)
            recon = self.decoder((z, y))
            if train_classifier:
                return mu, logvar, recon, c_out
            else:
                return mu, logvar, recon
        else:
            return c_out

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        # classifier_weights = OrderedDict()
        # decoder_weights = OrderedDict()
        
        # for k, v in zip(self.state_dict().keys(), weights):
        #     if k.startswith('classifier'):
                
        state_dict = OrderedDict(
            #{k.removeprefix("classifier.").removeprefix("decoder."): torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
            {k.lstrip("classifier").lstrip("decoder").lstrip("."): torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.classifier.load_state_dict(state_dict, strict=False)
        self.decoder.load_state_dict(state_dict, strict=False)
