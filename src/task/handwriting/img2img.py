import cv2
import numpy as np
import progressbar
import torch
import torch.nn as nn
from torch.nn import DataParallel

from src.encoding.PositionalEncoding import PositionalEncoding
from src.model.handwriting.image_encoder import ImageEncoder
from src.model.handwriting.image_decoder import ImageDecoder
from src.task.task import Task


class Img2Img_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Img2Img_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        pos_encoding = PositionalEncoding(config.char_embedding_dim)

        self.image_encoder = DataParallel(ImageEncoder(pos_encoding, config))
        self.image_decoder = DataParallel(ImageDecoder(config))

    def forward(self, batch_output):

        batch_output.device = self.device

        try:
            img2img_enc_char = self.image_encoder(batch_output['img_img'])
            img2img_dec_img = self.image_decoder(img2img_enc_char)

            return img2img_dec_img

        except RuntimeError:
            raise RuntimeWarning("Data is unavailable for img2img task.")

    def evaluate(self, batch_output):

        batch_output.device = self.device

        try:
            img2img_enc_char = self.image_encoder(batch_output['img_img'])
            img2img_dec_img = self.image_decoder(img2img_enc_char)

            return img2img_dec_img

        except RuntimeError:
            raise RuntimeWarning("Data is unavailable for img2img task.")

    def generate(self, batch_output):
        return self.evaluate(batch_output)


class Img2Img(Task):
    def __init__(self, train_set, val_set, test_set, gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, gen_set, char_model, config, device, exp_track)

    def build_model(self):
        return Img2Img_Model(self.char_model, self.config, self.device)

    def loss_function(self):
        return nn.MSELoss()

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_model(self):

        for epoch in range(self.current_epoch, self.config.epoch + 1):

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")

            self.model.train()
            self.current_epoch = epoch

            total_loss = []

            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    img2img_dec_img = self.model(batch_output)

                    img_loss = self.criterion(img2img_dec_img, batch_output['img_img'])

                    loss = img_loss

                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()

                    total_loss.append(loss.item())

                    bar.update(index)

            print("train_total_loss", np.average(total_loss))

            if epoch % self.config.model_eval_epoch == 0:

                self.exp_track.log_metric("train_total_loss", np.average(total_loss))

                self.eval_model()
                self.save_model()
                self.gen_model()

    def eval_model(self):

        self.model.eval()

        total_loss = []

        with torch.no_grad():

            print("\nEvaluating..............")
            with progressbar.ProgressBar(max_value=len(self.val_set)) as bar:

                for index, batch_output in enumerate(self.val_set):

                    try:
                        img2img_dec_img = self.model.evaluate(batch_output)
                        img2img_loss = self.criterion(img2img_dec_img, batch_output['img_img'])
                        total_loss.append(img2img_loss.item())

                    except RuntimeWarning:
                        pass

                    bar.update(index)

        self.exp_track.log_metric("val_total_loss", np.average(total_loss))

    def gen_model(self):

        def save_img(img_path, img, img_title):
            cv2.imwrite(img_path, img)
            self.exp_track.log_image(img_title, img)

        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():

            for index, batch_output in enumerate(self.gen_set, 1):

                if batch_output['label'] == 1:

                    output = self.model.generate(batch_output)

                    save_img(f"{self.config.gen_epoch_path}/img2img_in_{index}.png",
                             255 - batch_output['img_img'].squeeze(0).squeeze(0).cpu().numpy(), 'real_img2img')

                    save_img(f"{self.config.gen_epoch_path}/img2img_out_{index}.png",
                             255 - output.squeeze(0).squeeze(0).cpu().numpy(), 'predicted_img2img')

    def test_model(self):
        pass
