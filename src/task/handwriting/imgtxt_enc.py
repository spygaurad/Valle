import numpy as np
import progressbar
import torch
import torch.nn as nn
from torch.nn import DataParallel
from transformers import ReformerModel

from src.encoding.PositionalEncoding import PositionalEncoding
from src.model.handwriting.image_encoder import ImageEncoder
from src.model.handwriting.text_encoder import TextEncoder
from src.task.task import Task


class ImgTxt_Enc_Model(nn.Module):

    def __init__(self, char_model, config, device):
        super(ImgTxt_Enc_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        self.reformer_model = ReformerModel.from_pretrained("google/reformer-enwik8")

        for param in self.reformer_model.parameters():
            param.requires_grad = False

        pos_encoding = PositionalEncoding(config.char_embedding_dim)

        self.text_encoder = DataParallel(TextEncoder(config))
        self.image_encoder = DataParallel(ImageEncoder(pos_encoding, config))

    def forward(self, batch_output):

        batch_output.device = self.device
        self.reformer_model = self.reformer_model.eval()
        self.reformer_model = self.reformer_model.to(self.device)

        img_enc_txt = self.image_encoder(batch_output['img_txt_img'])

        txt_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                        batch_output['img_txt_txt_pad_mask'],
                                        self.reformer_model)

        return img_enc_txt, txt_enc_txt

    def evaluate(self, batch_output):

        batch_output.device = self.device
        self.reformer_model = self.reformer_model.eval()
        self.reformer_model = self.reformer_model.to(self.device)

        try:
            img_enc_txt = self.image_encoder(batch_output['img_txt_img'])

            txt_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                            batch_output['img_txt_txt_pad_mask'],
                                            self.reformer_model)

            return img_enc_txt, txt_enc_txt

        except RuntimeError:
            raise RuntimeWarning("Data is unavailable.")


class ImgTxt_Enc(Task):

    def __init__(self, train_set, val_set, test_set, gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, gen_set, char_model, config, device, exp_track)

    def build_model(self):
        return ImgTxt_Enc_Model(self.char_model, self.config, self.device)

    def loss_function(self):
        return nn.CosineSimilarity(dim=-1)

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_model(self):

        for epoch in range(self.current_epoch, self.config.epoch + 1):

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")

            self.model.train()
            self.current_epoch = epoch

            total_cos_emb_loss = []
            total_mae_loss = []
            total_cos_sim = []

            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    txt_enc_txt, img_enc_txt = self.model(batch_output)

                    loss = 1 - torch.mean(self.criterion(img_enc_txt, txt_enc_txt))

                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()

                    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)(img_enc_txt, txt_enc_txt)
                    mae_loss = nn.L1Loss()(img_enc_txt, txt_enc_txt)

                    total_cos_emb_loss.append(loss.item())
                    total_mae_loss.append(mae_loss.item())
                    total_cos_sim.append(torch.mean(cosine_similarity).item())

                    bar.update(index)

            print("train_total_cos_loss", np.average(total_cos_emb_loss))
            print("train_total_mae_loss", np.average(total_mae_loss))
            print("train_total_cos_sim", np.average(total_cos_sim))

            if epoch % self.config.model_eval_epoch == 0:

                self.exp_track.log_metric("train_total_cos_loss", np.average(total_cos_emb_loss))
                self.exp_track.log_metric("train_total_mae_loss", np.average(total_mae_loss))
                self.exp_track.log_metric("train_total_cos_sim", np.average(total_cos_sim))

                self.eval_model()
                self.save_model()

    def eval_model(self):

        self.model.eval()

        total_cos_emb_loss = []
        total_mae_loss = []
        total_cos_sim = []

        with torch.no_grad():

            print("\nEvaluating..............")
            with progressbar.ProgressBar(max_value=len(self.val_set)) as bar:

                for index, batch_output in enumerate(self.val_set):

                    try:
                        txt_enc_txt, img_enc_txt = self.model.evaluate(batch_output)
                        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)(img_enc_txt, txt_enc_txt)

                        loss = 1 - torch.mean(self.criterion(img_enc_txt, txt_enc_txt))
                        mae_loss = nn.L1Loss()(img_enc_txt, txt_enc_txt)

                        total_cos_emb_loss.append(loss.item())
                        total_mae_loss.append(mae_loss.item())
                        total_cos_sim.append(torch.mean(cosine_similarity).item())

                    except RuntimeWarning:
                        pass

                    bar.update(index)

        self.exp_track.log_metric("val_total_cos_loss", np.average(total_cos_emb_loss))
        self.exp_track.log_metric("val_total_mae_loss", np.average(total_mae_loss))
        self.exp_track.log_metric("val_total_cos_sim", np.average(total_cos_sim))

    def gen_model(self):
        pass

    def test_model(self):
        pass
