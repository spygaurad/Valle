import cv2
from neptune.new.types import File
import numpy as np
from pathlib import Path
import progressbar
import torch
from torch import autograd
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from transformers import ReformerModel

import os
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

plt.switch_backend('agg')

from src.encoding.PositionalEncoding import PositionalEncoding, LearnablePositionalEncoding
from src.model.handwriting.image_decoder import ImageDecoder
from src.model.handwriting.text_encoder import TextEncoder
from src.task.task import Task

class Txt2Img_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Txt2Img_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        self.reformer_model = ReformerModel.from_pretrained("google/reformer-enwik8")

        for param in self.reformer_model.parameters():
            param.requires_grad = False

        self.reformer_model = DataParallel(self.reformer_model)

        char_embedding = nn.Embedding(char_model.n_chars, config.char_embedding_dim)
        # pos_encoding = PositionalEncoding(config.char_embedding_dim)
        pos_encoding = PositionalEncoding(2048)
        # pos_encoding = LearnablePositionalEncoding(512, 128)

        self.text_encoder = DataParallel(TextEncoder(config))
        # self.image_decoder = DataParallel(ImageDecoder(config))
        self.image_decoder = DataParallel(ImageDecoder(char_model.n_chars, char_embedding, pos_encoding, config, device))

    def forward(self, batch_output):

        batch_output.device = self.device

        self.reformer_model = self.reformer_model.eval()
        self.reformer_model = self.reformer_model.to(self.device)

        # Text to Image
        txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                            batch_output['img_txt_txt_pad_mask'],self.reformer_model)

        # txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

        txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

        return txt2img_dec_img

    def evaluate(self, batch_output):

        batch_output.device = self.device
        self.reformer_model = self.reformer_model.eval()
        self.reformer_model = self.reformer_model.to(self.device)

        try:
            txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                batch_output['img_txt_txt_pad_mask'], self.reformer_model)

            # txt2img_dec_img = self.image_decoder(txt2img_enc_txt)
            txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

            return txt2img_dec_img

        except RuntimeError:
            raise RuntimeWarning("Data is unavailable for txt2img task.")

    def generate(self, batch_output):
        return self.evaluate(batch_output)


import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            #GPUtil.showUtilization()

            GPUs = GPUtil.getGPUs()
            import os
            device_id = int(os.getenv('CUDA_VISIBLE_DEVICES'))
            # device_id = torch.cuda.current_device()
            print("Current Device:", device_id)
            current_gpu = GPUs[device_id]
            print(' ID | GPU  | MEM')
            print('--------------')
            print(' {0:2d} | {1:3.0f}% | {2:3.0f}%'.format(current_gpu.id, current_gpu.load*100, current_gpu.memoryUtil*100))

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


class Txt2Img(Task):

    def __init__(self, train_set, val_set, test_set, gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, gen_set, char_model, config, device, exp_track)
        # self.train_batch_size = 8
        # self.val_batch_size = 8
        self.train_batch_size = self.config.img_txt_batch_size
        self.val_batch_size = self.config.img_txt_batch_size 
        self.writer = SummaryWriter(self.config.tensorboard_path)
        print("Train batch size: ", self.train_batch_size)
        print("Val batch size: ", self.val_batch_size)

    def build_model(self):
        return Txt2Img_Model(self.char_model, self.config, self.device)

    def loss_function(self):
        return nn.MSELoss(reduction="sum")
    
    def get_scheduler(self):
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 8000, eta_min=0.0001/2.0)
        return scheduler

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        # return torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def get_scaler(self):
        # scaler = torch.cuda.amp.GradScaler(init_scale=0.125)
        # scaler = torch.cuda.amp.GradScaler()
        scaler = None
        return scaler

    def get_grad_flow(self, named_parameters):
	    ave_grads = []
	    max_grads = []
	    layers = []
	    for n, p in named_parameters:
	        if (p.requires_grad) and ("bias" not in n):
                    if p.grad is not None:
                        layers.append(n)
                        ave_grads.append(torch.log(p.grad.abs().mean()))
                        # ave_grads.append(p.grad)
			# max_grads.append(p.grad.abs().max())
	    print('\n')
	    # print(layers)
	    print('\n')
	    print(ave_grads)
	    # print(max_grads)
	    print('\n\n\n')

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Conv1d:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


    def train_model(self):

        print("************************************************")
        # self.eval_model()
        # self.gen_model()
        print("************************************************")
        # scaler = torch.cuda.amp.GradScaler(init_scale=65536.0/65536)

        # monitor = Monitor(60)

        # self.model.apply(self.init_weights)

        for epoch in range(self.current_epoch, self.config.epoch + 1):

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")
            # print("Scalar Stat:", self.scaler.state_dict())
            for param_group in self.optimizer.param_groups:
                # if epoch > 20:
                #     param_group['lr'] = 0.001
                current_lr = param_group['lr']
                print("Current learning rate is: {}".format(current_lr))

            self.model.train()
            self.current_epoch = epoch

            total_loss = []

            accumulation_step = 3
            self.optimizer.zero_grad()

            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:


                for index, batch_output in enumerate(self.train_set):

                    # with autograd.detect_anomaly():

                    self.optimizer.zero_grad()

                    txt2img_dec_img = self.model(batch_output)

                    txt2img_loss = self.criterion(txt2img_dec_img, batch_output['img_txt_img'])

                    self.optimizer.zero_grad()
                    loss = txt2img_loss
                    loss = loss / (self.config.img_txt_batch_size)
                    loss.backward()

                    self.optimizer.step()

                    # # self.get_grad_flow(self.model.named_parameters())
                    # total_loss.append(loss.item())
                    # print(batch_output['img_txt_img'].is_pinned())

                    # with autograd.detect_anomaly():

                    # with torch.cuda.amp.autocast():
                    #     txt2img_dec_img = self.model(batch_output)
                    #     output = batch_output['img_txt_img']
                    #     # print(txt2img_dec_img.shape)
                    #     # print(output.shape)
                    #     txt2img_loss = self.criterion(txt2img_dec_img, output)
                    #     # txt2img_loss = txt2img_loss / (self.config.img_txt_batch_size)
                    #     loss = txt2img_loss / self.train_batch_size
                    #     # loss = loss / self.train_batch_size
                    # self.scaler.scale(loss).backward()
                    # # self.get_grad_flow(self.model.named_parameters())
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    total_loss.append(loss.item())


                    bar.update(index)
                    # scheduler.step()
                    # if epoch <= 10:
                    #     scheduler.step()




            # scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 20, eta_min=0.0001/2.0)
            # self.scheduler.step()
            # if epoch <= 10:
            #     scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, steps, eta_min = 0.0005)

            train_loss = np.average(total_loss)
            print("train_total_loss", train_loss)

            self.results["epochs"].append(epoch)
            self.results["train_loss"].append(train_loss)

            # val_loss = self.eval_model()
            # self.results["val_loss"].append(val_loss)


            # self.plot()

            self.writer.add_scalars('Loss', {'train_loss': train_loss}, epoch)
            self.writer.add_scalar('LR Schedule', current_lr, epoch)

            if epoch % self.config.model_eval_epoch == 0:

                if self.exp_track is not None:
                    self.exp_track["train_total_loss"].log(train_loss)

                val_loss = self.eval_model()
                self.results["val_loss"].append(val_loss)
                self.writer.add_scalars('Loss', {'val_loss':val_loss}, epoch)
                # self.save_model()
                self.gen_model()

        # monitor.stop()

    def plot(self):
        fig = plt.figure()
        plt.plot(self.results["epochs"], self.results["train_loss"], label="train")
        plt.plot(self.results["epochs"], self.results["val_loss"], label="val")
        plt.title("Loss of txt2img")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        self.writer.add_figure('Loss', fig)

        fig_savepath = self.config.EXP_DIR/self.config.run_id/'loss.png'
        fig.savefig(fig_savepath)
        plt.show()
        plt.close(fig)

    def eval_model(self):

        self.model.eval()

        # train_total_loss = []
        # with torch.no_grad():

        #     print("\nEvaluating..............")
        #     with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

        #         for index, batch_output in enumerate(self.train_set):

        #             try:
        #                 with torch.cuda.amp.autocast():
        #                     txt2img_dec_img = self.model.evaluate(batch_output)
        #                     txt2img_loss = self.criterion(txt2img_dec_img, batch_output['img_txt_img'])
        #                     txt2img_loss = txt2img_loss / self.val_batch_size
        #                 train_total_loss.append(txt2img_loss.item())
        #             except RuntimeWarning:
        #                 pass
        #             bar.update(index)
        # print("train_total_loss", np.average(train_total_loss))

        total_loss = []

        with torch.no_grad():

            print("\nEvaluating..............")
            with progressbar.ProgressBar(max_value=len(self.val_set)) as bar:

                for index, batch_output in enumerate(self.val_set):

                    try:
                        # with torch.cuda.amp.autocast():
                        txt2img_dec_img = self.model.evaluate(batch_output)
                        txt2img_loss = self.criterion(txt2img_dec_img, batch_output['img_txt_img'])
                        txt2img_loss = txt2img_loss / self.val_batch_size
                        total_loss.append(txt2img_loss.item())
                    except RuntimeWarning:
                        pass

                    bar.update(index)

        val_loss = np.average(total_loss)

        if self.exp_track is not None:
            self.exp_track["val_total_loss"].log(val_loss)

        print("val_total_loss", val_loss)

        return val_loss

    def gen_model(self):

        def save_img(img_path, img, img_title):
            print(img_path)
            cv2.imwrite(img_path, img)
            save_path = str(Path(*Path(img_path).parts[-3:]))
            if self.exp_track is not None:
                self.exp_track[save_path].upload(File.as_image(img.squeeze()))

        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            print("Val Set")

            for index, batch_output in enumerate(self.gen_set, 1):

                if batch_output['label'] == 2:

                    # output = self.model.generate(batch_output)
                    output = self.model(batch_output)

                    save_img(f"{self.config.gen_epoch_path}/val_txt2img_in_{index}.png",
                             (255) * (batch_output['img_txt_img'].squeeze(0).squeeze(0).cpu().numpy()), 'real_txt2img')

                    save_img(f"{self.config.gen_epoch_path}/val_txt2img_out_{index}.png",
                             (255) * (output.squeeze(0).squeeze(0).cpu().numpy()), 'predicted_txt2img')

                    # save_img(f"{self.config.gen_epoch_path}/val_txt2img_in_{index}.png",
                    #          (255) * ((batch_output['img_txt_img'].squeeze(0).squeeze(0).cpu().numpy()) * 0.5 + 0.5), 'real_txt2img')

                    # save_img(f"{self.config.gen_epoch_path}/val_txt2img_out_{index}.png",
                    #          (255) * ((output.squeeze(0).squeeze(0).cpu().numpy()) * 0.5 + 0.5), 'predicted_txt2img')
        counter = 1
        with torch.no_grad():
            print("Train Set")

            for index, batch_output in enumerate(self.test_set, 1):

                if batch_output['label'] == 2:

                    output = self.model(batch_output)

                    save_img(f"{self.config.gen_epoch_path}/train_txt2img_in_{index}.png",
                             (255) * (batch_output['img_txt_img'].squeeze(0).squeeze(0).cpu().numpy()), 'real_txt2img')

                    save_img(f"{self.config.gen_epoch_path}/train_txt2img_out_{index}.png",
                             (255) * (output.squeeze(0).squeeze(0).cpu().numpy()), 'predicted_txt2img')

                    # save_img(f"{self.config.gen_epoch_path}/train_txt2img_in_{index}.png",
                    #          (255) * ((batch_output['img_txt_img'].squeeze(0).squeeze(0).cpu().numpy()) * 0.5 + 0.5), 'real_txt2img')

                    # save_img(f"{self.config.gen_epoch_path}/train_txt2img_out_{index}.png",
                    #         (255) * ((output.squeeze(0).squeeze(0).cpu().numpy()) * 0.5 + 0.5), 'predicted_txt2img')
                    counter += 1
                    print(counter)
                if (counter % 20) == 0:
                    break

    def test_model(self):
        pass
