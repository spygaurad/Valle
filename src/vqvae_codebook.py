from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import glob
from six.moves import xrange

import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import cv2
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps, ImageFont
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import Augmentor
from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)


def create_img(text, font):
    generator = GeneratorFromStrings(
        [text,],
        blur=1,
        random_blur=False,
        fonts = [font,],
        count = 1,
        size = 64,
        width = 1024,
        background_type = 1
    )
    img, label = list(generator)[0]
    img = ImageOps.grayscale(img)

    if np.random.random() < 0.5:
        p = Augmentor.Pipeline()
        p.gaussian_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=6, corner="dl", method="out")
        transforms = torchvision.transforms.Compose([p.torch_transform(), torchvision.transforms.GaussianBlur(5, sigma=(0.1, 0.1)), torchvision.transforms.RandomAdjustSharpness(2, p=1.0)])
        img = transforms(img.copy())

    # img = Image.new("RGB", (1024, 64), (255, 255, 255))
    # draw = ImageDraw.Draw(img)

    # width, height = img.size

    # textwidth, textheight = draw.textsize(text)

    # if font=="roboto_mono":
    #     font="yanone"

    # # 512->8, 1024-> 16
    # # font_path = f"fonts/{font}"
    # font_path = font
    # font = ImageFont.truetype(font_path, 30)

    # x = 5
    # y = (height/2) - textheight

    # draw.text((x, y), text, (0,0,0), font=font)
    img = np.array(ImageOps.grayscale(img))

    return img

class ImgTxtDataset(Dataset):
    def __init__(self, data, fonts):
        super().__init__()
        self.data = data
        self.fonts = fonts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        font_type = np.random.choice(self.fonts)

        img = 255 - create_img(text, font_type)
        img = img/255

        dataset_instance = {
            "img": torch.from_numpy(img[np.newaxis]).to(torch.float)
            }

        return dataset_instance

class HandwritingDataloader:
    def __init__(self, df_file_path):
        # self.img_txt_data = pd.read_csv(df_file_path, on_bad_lines='skip').sample(frac=1)
        self.img_txt_data = pd.read_csv(df_file_path, sep="\t", names=["text"], encoding="utf-8")
        # self.img_txt_data = pd.read_csv(df_file_path, delimeter=None).sample(frac=1)


    def split_data(self, dataset):
        """
        train_len = 2
        test_len = 2
        val_len = 2
        val_test_len = 6
        val_gen_len = 4
        """
        # dataset_len = 5000
        dataset_len = len(dataset)
        # dataset_len = 1000 
        train_len = int(0.95 * dataset_len)
        val_test_len = dataset_len - train_len
        val_gen_len = int(0.5 * val_test_len)
        test_len = val_test_len - val_gen_len
        val_len = val_gen_len - 5
        #"""
        print(train_len, val_len, val_gen_len, test_len)

        train_set, val_test = random_split(
            dataset,
            [train_len, val_test_len],
        )

        val_gen_set, test_set = random_split(
            val_test,
            [val_gen_len, test_len],
        )

        val_set, gen_set = random_split(
            val_gen_set,
            [val_len, 5],
        )

        return train_set, val_set, test_set, gen_set

    def load_data(self, batch_size, fonts):
        batch_size = batch_size
        shuffle = True
        num_workers = 2
        img_txt_dataset = ImgTxtDataset(self.img_txt_data[:], fonts)

        train_set, val_set, test_set, gen_set = self.split_data(img_txt_dataset)

        # data_variance = np.var(train_set.data)
        data_variance = None

        train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=num_workers,
                                      )

        val_dataloader = DataLoader(val_set, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers,
                                    )

        test_dataloader = DataLoader(test_set, batch_size=1,
                                     shuffle=shuffle, num_workers=num_workers,
                                     )

        val_gen_dataloader = DataLoader(val_set, batch_size=1,
                                    shuffle=shuffle, num_workers=num_workers,
                                    )

        train_gen_dataloader = DataLoader(train_set, batch_size=1,
                                    shuffle=shuffle, num_workers=num_workers,
                                    )

        return train_dataloader, val_dataloader, test_dataloader, train_gen_dataloader, val_gen_dataloader, data_variance


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        # self._ema_cluster_size = torch.zeros(1, num_embeddings)
        
        self._decay = decay
        self._epsilon = epsilon

    def get_encodings_embeddings(self, inputs=None, encodings=None):

        if encodings is None:
            # convert inputs from BCHW -> BHWC
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            input_shape = inputs.shape
            
            # Flatten input
            flat_input = inputs.view(-1, self._embedding_dim)
            
            # Calculate distances
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(self._embedding.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
                
            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            # Quantize and unflatten
            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
            b,h,w,c = input_shape

            return encoding_indices.view(b, -1), quantized.permute(0, 3, 1, 2).contiguous()
        else:
            bs,hw = encodings.shape
            h = 4
            w = int(hw/h)
            c = self._embedding_dim

            encoding_indices = encodings.view(-1, 1)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            # Quantize and unflatten
            quantized = torch.matmul(encodings, self._embedding.weight).view(bs, h, w, c)
            # convert quantized from BHWC -> BCHW
            return quantized.permute(0, 3, 1, 2).contiguous()


    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            """
            print("inputs:", input_shape)
            print("flat_input:", flat_input.shape)
            print("distances:", distances.shape)
            print("encoding indicies:", encoding_indices.shape)
            print("Number of embeddings:", self._num_embeddings)
            print("Embedding dim:", self._embedding_dim)
            print("Encodings shape:", encodings.shape)
            print("EMA cluster shape:", self._ema_cluster_size.shape)
            print("unsqueeze shape:", self._ema_cluster_size.unsqueeze(1).shape)
            print("ema shape: ", self._ema_w.shape)
            print("Embedding:", self._embedding.weight.shape)
            print("***", self._ema_cluster_size)
            print("\n")
            """



            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=1,
                                 out_channels=4,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._residual_stack_1 = ResidualStack(in_channels=4,
                                             num_hiddens=4,
                                             num_residual_layers=2,
                                             num_residual_hiddens=256)

        self._conv_2 = nn.Conv2d(in_channels=4,
                                 out_channels=16,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._residual_stack_2 = ResidualStack(in_channels=16,
                                             num_hiddens=16,
                                             num_residual_layers=2,
                                             num_residual_hiddens=256)

        self._conv_3 = nn.Conv2d(in_channels=16,
                                 out_channels=32,
                                 kernel_size=3,
                                 stride=2, padding=1)

        self._residual_stack_3 = ResidualStack(in_channels=32,
                                             num_hiddens=32,
                                             num_residual_layers=2,
                                             num_residual_hiddens=256)

        self._conv_4 = nn.Conv2d(in_channels=32,
                                 out_channels=64,
                                 kernel_size=3,
                                 stride=(2,1), padding=1)

        self._residual_stack_4 = ResidualStack(in_channels=64,
                                             num_hiddens=64,
                                             num_residual_layers=2,
                                             num_residual_hiddens=256)

        self._conv_5 = nn.Conv2d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=3,
                                 stride=(1,1), padding=1)

        self._residual_stack_5 = ResidualStack(in_channels=128,
                                             num_hiddens=128,
                                             num_residual_layers=2,
                                             num_residual_hiddens=256)

        self._conv_6 = nn.Conv2d(in_channels=128,
                                 out_channels=256,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=256,
                                             num_hiddens=256,
                                             num_residual_layers=2,
                                             num_residual_hiddens=256)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        # x = F.relu(x)
        x = self._residual_stack_1(x)
        
        x = self._conv_2(x)
        # x = F.relu(x)
        x = self._residual_stack_2(x)
        
        x = self._conv_3(x)
        # x = F.relu(x)
        x = self._residual_stack_3(x)

        x= self._conv_4(x)
        # x = F.relu(x)
        x = self._residual_stack_4(x)

        x= self._conv_5(x)
        # x = F.relu(x)
        x = self._residual_stack_5(x)

        x= self._conv_6(x)

        return self._residual_stack(x)



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=256,
                                 out_channels=128,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=128,
                                             num_hiddens=128,
                                             num_residual_layers=2,
                                             num_residual_hiddens=128)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=128, 
                                                out_channels=64,
                                                kernel_size=2, 
                                                stride=2, padding=0)
        
        self._residual_stack_1 = ResidualStack(in_channels=64,
                                             num_hiddens=64,
                                             num_residual_layers=2,
                                             num_residual_hiddens=64)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=64, 
                                                out_channels=32,
                                                kernel_size=2, 
                                                stride=2, padding=0)
        
        self._residual_stack_2 = ResidualStack(in_channels=32,
                                             num_hiddens=32,
                                             num_residual_layers=2,
                                             num_residual_hiddens=32)
        
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=32, 
                                                out_channels=16,
                                                kernel_size=2, 
                                                stride=2, padding=0)

        self._residual_stack_3 = ResidualStack(in_channels=16,
                                             num_hiddens=16,
                                             num_residual_layers=2,
                                             num_residual_hiddens=16)

        self._conv_trans_4 = nn.ConvTranspose2d(in_channels=16, 
                                                out_channels=4,
                                                kernel_size=(2,3), 
                                                stride=(2,1), padding=(0,1))

        self._residual_stack_4 = ResidualStack(in_channels=4,
                                             num_hiddens=4,
                                             num_residual_layers=2,
                                             num_residual_hiddens=4)

        self._conv_trans_5 = nn.ConvTranspose2d(in_channels=4, 
                                                out_channels=1,
                                                kernel_size=(3,3), 
                                                stride=(1,1), padding=(1,1))

        self._residual_stack_5 = ResidualStack(in_channels=1,
                                             num_hiddens=1,
                                             num_residual_layers=2,
                                             num_residual_hiddens=1)

    def forward(self, inputs):
        # print(inputs.shape)
        x = self._conv_1(inputs)
        # print(x.shape)
        
        x = self._residual_stack(x)
        # print(x.shape)
        
        x = self._conv_trans_1(x)
        # print(x.shape)

        x = self._residual_stack_1(x)
        # print(x.shape)

        x = self._conv_trans_2(x)
        # print(x.shape)

        x = self._residual_stack_2(x)
        # print(x.shape)

        x = self._conv_trans_3(x)
        # print(x.shape)

        x = self._residual_stack_3(x)
        # print(x.shape)

        x = self._conv_trans_4(x)
        # print(x.shape)

        x = self._residual_stack_4(x)
        # print(x.shape)

        x = self._conv_trans_5(x)
        # print(x.shape)

        # x = self._residual_stack_5(x)

        
        return x


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder()
        # self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
        #                               out_channels=embedding_dim,
        #                               kernel_size=1, 
        #                               stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder()

    def forward(self, x):
        z = self._encoder(x)
        # z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        # print("Image:", x.shape)
        # print("Latent:", z.shape)
        # print("Reconstructed:", x_recon.shape)

        return loss, x_recon, perplexity

    def get_encodings(self, x):
        z = self._encoder(x)
        encodings, quantized = self._vq_vae.get_encodings_embeddings(inputs=z)
        return encodings, quantized

    def get_reconstructed_img(self, z, quantization=True):
        # print("Z:", z.shape)
        if quantization:
            quantized = z
        else:
            quantized = self._vq_vae.get_encodings_embeddings(encodings=z)
        x_recon = self._decoder(quantized)
        return x_recon


def get_model():
    embedding_dim = 256
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99

    model = Model(num_embeddings, embedding_dim, 
                  commitment_cost, decay)

    return model

def train_codebook(load_saved_model=False):

    def compute_stats(dataloader):
      model.eval()
      val_res_recon_error = []
      val_res_perplexity = []
      with torch.no_grad():
        for d in dataloader:
          data = d['img']
          data = data.to(device)
          vq_loss, data_recon, perplexity = model(data)
          recon_error = F.mse_loss(data_recon, data, reduction='sum') / batch_size

          val_res_recon_error.append(recon_error.item())
          val_res_perplexity.append(perplexity.item())

        print('Val recon_error: %.3f' % np.mean(val_res_recon_error[:]))
        print('Val perplexity: %.3f' % np.mean(val_res_perplexity[:]))

    def get_plot(dataloader, data_type="train", iterations=0):
      model.eval()

      out_img = []
      gap_img = np.zeros((8, 1024))

      with torch.no_grad(): 
          print(f"{data_type} Set")
          for index, data in enumerate(dataloader):
              img = data["img"].to(device)
            #   plt.figure(figsize=(40,8))
            #   plt.imshow(img.cpu().squeeze().numpy()*255, cmap="gray")

              # encoder = model._encoder(img)
              # _, quantized, _, _ = model._vq_vae(encoder)
              # reconstructions = model._decoder(quantized)

              vq_loss, reconstructions, perplexity = model(img)
            #   plt.figure(figsize=(30,8))
            #   plt.imshow(reconstructions.cpu().squeeze().numpy()*255, cmap="gray")

              out_img.append(img.cpu().squeeze().numpy()*255)
              out_img.append(gap_img)
              out_img.append(reconstructions.cpu().squeeze().numpy()*255)

              if index >= 15:
                  break

          cv2.imwrite(f"out/{data_type}_{iterations}.png", np.vstack(out_img))
          cv2.imwrite(f"{data_type}.png", np.vstack(out_img))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fonts = list(glob.glob("fonts/*.ttf"))
    print(np.random.choice(fonts))
    df_file_path = "./dataset.csv"
    batch_size = 16 #int(128/4) + 14
    train_set, val_set, test_set, train_gen_set, val_gen_set, data_variance = HandwritingDataloader(df_file_path).load_data(batch_size, fonts)

    num_training_updates = 60000 * 100
    start_updates = 0
    learning_rate = 1e-3

    model = get_model()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    training_loader = train_set
    validation_loader = val_set

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    if load_saved_model:
        checkpoint = torch.load("latest_model.pth")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_updates= checkpoint["iteration"] + 1
        print(f"Model Loaded: {start_updates}")

    compute_stats(validation_loader)
    print(f'Started training: {time.asctime()}')
    print("\n")
    for i in xrange(start_updates, num_training_updates):
        model.train()
        data  = next(iter(training_loader))
        data = data['img']
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)

        print("********************************************")
        print("Data Reconstruct:", data_recon.shape)
        print("********************************************")
        print("Encoding:", model.get_encodings(data)[1].shape)
        print("********************************************")
        print("Reconstructed:", model.get_reconstructed_img(model.get_encodings(data)[1]).shape)
        print("********************************************")
        break

        # print("********************************************")
        # print("Reconstructed:", model.get_reconstructed_img(model.get_encodings(data)).shape)
        # print("********************************************")
        # break

        vq_loss, data_recon, perplexity = model(data)
        # recon_error = F.mse_loss(data_recon, data) / data_variance
        recon_error = F.mse_loss(data_recon, data, reduction='sum') / batch_size
        loss = recon_error + vq_loss

        loss.backward()
        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i+1) % 100 == 0:
            print(f"Iteration: {i+1} -- {time.asctime()}")
            print('%d iterations' % (i+1))
            print('Train recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('Train perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            compute_stats(validation_loader)
            get_plot(train_gen_set, "train", i)
            get_plot(val_gen_set, "val", i)
            model_to_save = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': i}
            torch.save(model_to_save, "latest_model.pth")
            print()

if __name__ == "__main__":
    # train_codebook(False)
    train_codebook(True)
