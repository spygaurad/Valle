import cv2
import numpy as np
import pandas as pd
import progressbar
import re
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from torchmetrics.functional import char_error_rate, word_error_rate

import torchvision

from src.model.handwriting.style_encoder_decoder import StyleNet
from src.task.task import Task

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op
        print(self.optimizers)

    def __getitem__(self, key):
        return self.optimizers[key]

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dict):
        for index, op in enumerate(self.optimizers):
            op.load_state_dict(state_dict[index])



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Style_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Style_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device
        num_writers = 657


        # self.style_net = DataParallel(StyleNet(config, char_model, num_writers, device))
        self.style_net = (StyleNet(config, char_model, num_writers, device))

        print(f"Number of parameters in style net : {count_parameters(self.style_net)}")
        

    def forward(self, batch_output, out_only=False):

        batch_output.device = self.device
        # content_img, style_img, content_txt, content_txt_len, writers
        # ctc_loss = self.criterion['ctc'](aux_features, batch_output['img_txt_txt_tgt'], aux_len, batch_output['img_txt_txt_tgt_len']) 
        content_img = batch_output['img_txt_img']
        style_img = batch_output['img_txt_style_img']
        style_img_collection = batch_output['img_txt_style_img_collection']
        content_txt = batch_output['img_txt_txt_tgt']
        content_txt_len = batch_output['img_txt_txt_tgt_len']
        writers = batch_output['writer_id']
        return_dict = self.style_net(content_img, style_img, style_img_collection, content_txt, content_txt_len, writers, out_only)

        return return_dict

    def evaluate(self, batch_output, out_only=True):

        batch_output.device = self.device
        # content_img, style_img, content_txt, content_txt_len, writers
        content_img = batch_output['img_txt_img']
        style_img = batch_output['img_txt_style_img']
        style_img_collection = batch_output['img_txt_style_img_collection']
        content_txt = batch_output['img_txt_txt_tgt']
        content_txt_len = batch_output['img_txt_txt_tgt_len']
        writers = batch_output['writer_id']
        out_img = self.style_net(content_img, style_img, style_img_collection, content_txt, content_txt_len, writers, out_only)

        return out_img

class StyleTransfer(Task):

    def __init__(self, train_set, val_set, test_set, train_gen_set, val_gen_set, test_gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, val_gen_set, char_model, config, device, exp_track)

        self.train_gen_set = train_gen_set
        self.val_gen_set = val_gen_set
        self.test_gen_set = test_gen_set

        self.train_batch_size = self.config.batch_size
        self.val_batch_size = self.config.batch_size
        self.writer = SummaryWriter(self.config.tensorboard_path)

        print("StyleTransfer")

        print(f"Train Batch Size: {self.train_batch_size}")
        print(f"Val Batch Size: {self.val_batch_size}")

        self.use_scheduler = False
        self.warmup_epochs = 0

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def build_model(self):
        return Style_Model(self.char_model, self.config, self.device)

    def get_scheduler(self):
        # scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 500, eta_min=0.0005/10.0)
        scheduler = None
        return scheduler

    def get_scaler(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision_training)
        # scaler = None
        return scaler

    def loss_function(self):
        pass

    def get_optimizer(self):
        # return torch.optim.Adam(self.model.parameters(), lr=0.0001)
        encoder_optimizer = torch.optim.AdamW([{'params': self.model.style_net.content_encoder.parameters()}, {'params': self.model.style_net.style_encoder.parameters()}], lr=0.0001)
        decoder_optimizer = torch.optim.AdamW(self.model.style_net.decoder.parameters(), lr=0.0001)
        return MultipleOptimizer(encoder_optimizer, decoder_optimizer)

    def get_plot(self, dataloader_type="train", iterations=0):
        print(f"Plotting {dataloader_type}")
        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)
        outdir = self.config.gen_epoch_path

        if dataloader_type == "train":
            dataloader = self.train_gen_set
            data_type = "train"

        elif dataloader_type == "val":
            dataloader =self.val_gen_set
            data_type = "val"

        self.model.eval()

        out_img = []
        gap_img = np.zeros((8, 2048))

        with torch.no_grad(): 
            for index, data in enumerate(dataloader):
                data.device = self.device
                content_img = data["img_txt_img"]
                style_img = data["img_txt_style_img"]

                reconstructions = self.model.evaluate(data, out_only=True)

                out_img.append(content_img.cpu().squeeze().numpy() * 255)
                out_img.append(gap_img)
                out_img.append(style_img.cpu().squeeze().numpy() * 255)
                out_img.append(gap_img)
                out_img.append(reconstructions.cpu().squeeze().numpy() * 255)

                if index >= 10:
                    break

            outpath = str(outdir/f"{data_type}_{iterations+1}.png")
            cv2.imwrite(outpath, np.vstack(out_img))
        print(f"Plotting Completed")


    def store_char_seq_prediction(self, prediction, target, ctc_mode=True):
       target = target
       predicted = prediction
       bs = target.shape[0]
       token = "TEOS"
       store_list = []
       for i in range(bs):
           str_target = (self.char_model.indexes2characters(target[i].cpu().numpy()))
           if token in str_target:
               str_target_first_pad_index = str_target.index(token)
           else:
               str_target_first_pad_index = len(str_target)

           str_target = "".join(str_target[:str_target_first_pad_index])

           str_predicted =  (self.char_model.indexes2characters(predicted[i].cpu().numpy(), ctc_mode=True))
           if token in str_predicted:
               str_predicted_first_pad_index = str_predicted.index(token)
           else:
               if "PAD" in str_predicted:
                   str_predicted_first_pad_index = str_predicted.index("PAD")
               else:
                   str_predicted_first_pad_index = len(str_predicted)
           str_predicted = "".join(str_predicted[:str_predicted_first_pad_index])

           if ctc_mode:
               if str_target.startswith("TSOS"):
                   str_target = str_target[4:]
               if str_predicted.startswith("TSOS"):
                   str_predicted = str_predicted[4:]

           store_list.append((str_target, str_predicted))

       return store_list

    def store_writers_prediction(self, prediction, target):
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()
        store_list = list(zip(target, prediction))
        return store_list

    def train_model(self):
        # print(self.model)

        print("Style task")

        # self.eval_model()

        for epoch in range(self.current_epoch, self.config.epoch + 1):

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")

            # for param_group in self.optimizer.param_groups:

            #     if epoch <= self.warmup_epochs:
            #         param_group['lr'] = (epoch/self.warmup_epochs) * self.config.lr

            #     # param_group['lr'] = self.config.lr/10

            #     current_lr = param_group['lr']
            #     # print(f"Param Group: {param_group}")
            #     print(f"Current LR: {current_lr}")

            self.model.train()
            self.current_epoch = epoch

            total_loss = []
            total_loss_ctc = []
            total_loss_ce = []
            total_loss_c = []
            total_loss_s = []
            total_loss_ctc_out = []
            total_loss_ce_out = []
            total_loss_auto = []
            train_char_seq_prediction = []
            train_writer_prediction = []
            train_out_char_seq_prediction = []
            train_out_writer_prediction = []
            print("Mixed precision:", self.use_mixed_precision_training)



            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    if epoch <= 1000:
                        self.optimizer[0].zero_grad()
                        self.set_requires_grad(self.model.style_net.decoder, False)
                        return_dict = self.model(batch_output)

                        predicted_char_seq = return_dict["outputs"]["predicted_char_seq"]
                        predicted_writers = return_dict["outputs"]["predicted_writers"]
                        out_predicted_char_seq = return_dict["outputs"]["out_predicted_char_seq"]
                        out_predicted_writers = return_dict["outputs"]["out_predicted_writers"]
                        out_img = return_dict["outputs"]["out_img"]

                        loss_ctc = return_dict["losses"]["loss_ctc"]
                        loss_ce = return_dict["losses"]["loss_ce"]
                        loss_c = return_dict["losses"]["loss_content"]
                        loss_s = return_dict["losses"]["loss_style"]
                        loss_ctc_out = return_dict["losses"]["out_loss_ctc"]
                        loss_ce_out = return_dict["losses"]["out_loss_ce"]
                        loss_auto = return_dict["losses"]["loss_auto"]

                        encoder_loss = 50 * loss_ctc + 1000 * loss_ce
                        encoder_loss.backward()
                        self.optimizer[0].step()
                        self.optimizer[0].zero_grad()
                        self.set_requires_grad(self.model.style_net.decoder, True)

                    if epoch > 1000:
                        self.optimizer[1].zero_grad()
                        self.set_requires_grad(self.model.style_net.style_encoder, False)
                        self.set_requires_grad(self.model.style_net.content_encoder, False)
                        return_dict = self.model(batch_output)

                        predicted_char_seq = return_dict["outputs"]["predicted_char_seq"]
                        predicted_writers = return_dict["outputs"]["predicted_writers"]
                        out_predicted_char_seq = return_dict["outputs"]["out_predicted_char_seq"]
                        out_predicted_writers = return_dict["outputs"]["out_predicted_writers"]
                        out_img = return_dict["outputs"]["out_img"]

                        loss_ctc = return_dict["losses"]["loss_ctc"]
                        loss_ce = return_dict["losses"]["loss_ce"]
                        loss_c = return_dict["losses"]["loss_content"]
                        loss_s = return_dict["losses"]["loss_style"]
                        loss_ctc_out = return_dict["losses"]["out_loss_ctc"]
                        loss_ce_out = return_dict["losses"]["out_loss_ce"]
                        loss_auto = return_dict["losses"]["loss_auto"]

                        decoder_loss = 50 * loss_ctc_out + 1000 * loss_ce_out + loss_auto + loss_c + 100 * loss_s
                        decoder_loss.backward()
                        self.optimizer[1].step()
                        self.optimizer[1].zero_grad()
                        self.set_requires_grad(self.model.style_net.style_encoder, True)
                        self.set_requires_grad(self.model.style_net.content_encoder, True)


                    loss = loss_ctc + loss_ce + loss_c + loss_s + loss_ctc_out + loss_ce_out + loss_auto
                    loss = loss.detach()
                    # loss.backward()

                    total_loss.append(loss.item())
                    total_loss_ctc.append(loss_ctc.item())
                    total_loss_ce.append(loss_ce.item())
                    total_loss_c.append(loss_c.item())
                    total_loss_s.append(loss_s.item())
                    total_loss_ctc_out.append(loss_ctc_out.item())
                    total_loss_ce_out.append(loss_ce_out.item())
                    total_loss_auto.append(loss_auto.item())

                    train_char_seq_prediction.extend(self.store_char_seq_prediction(predicted_char_seq, batch_output["img_txt_txt_tgt"]))
                    train_out_char_seq_prediction.extend(self.store_char_seq_prediction(out_predicted_char_seq, batch_output["img_txt_txt_tgt"]))
                    train_writer_prediction.extend(self.store_writers_prediction(predicted_writers, batch_output['writer_id']))
                    train_out_writer_prediction.extend(self.store_writers_prediction(out_predicted_writers, batch_output['writer_id']))

                    bar.update(index)

            if self.use_scheduler:
                self.scheduler.step()

            train_loss = np.average(total_loss)
            train_loss_ctc = np.average(total_loss_ctc)
            train_loss_ce = np.average(total_loss_ce)
            train_loss_c = np.average(total_loss_c)
            train_loss_s = np.average(total_loss_s)
            train_loss_ctc_out = np.average(total_loss_ctc_out)
            train_loss_ce_out = np.average(total_loss_ce_out)
            train_loss_auto = np.average(total_loss_auto)

            train_char_seq_prediction_df = pd.DataFrame(train_char_seq_prediction, columns=["target", "predicted"])
            train_df = train_char_seq_prediction_df
            train_cer = char_error_rate(preds=train_df.predicted.values.tolist(), target=train_df.target.values.tolist())
            train_wer = word_error_rate(preds=train_df.predicted.values.tolist(), target=train_df.target.values.tolist())
            train_acc = (train_df["target"] == train_df["predicted"]).sum()/train_df.shape[0]

            train_writer_prediction_df = pd.DataFrame(train_writer_prediction, columns=["target", "predicted"])
            train_writer_acc = (train_writer_prediction_df["target"] == train_writer_prediction_df["predicted"]).sum()/train_writer_prediction_df.shape[0]

            train_out_char_seq_prediction_df = pd.DataFrame(train_out_char_seq_prediction, columns=["target", "predicted"])
            train_out_df = train_out_char_seq_prediction_df
            train_out_cer = char_error_rate(preds=train_out_df.predicted.values.tolist(), target=train_out_df.target.values.tolist())
            train_out_wer = word_error_rate(preds=train_out_df.predicted.values.tolist(), target=train_out_df.target.values.tolist())
            train_out_acc = (train_out_df["target"] == train_out_df["predicted"]).sum()/train_out_df.shape[0]

            train_out_writer_prediction_df = pd.DataFrame(train_out_writer_prediction, columns=["target", "predicted"])
            train_out_writer_acc = (train_out_writer_prediction_df["target"] == train_out_writer_prediction_df["predicted"]).sum()/train_out_writer_prediction_df.shape[0]

            # print(train_char_seq_prediction_df)
            # print(train_writer_prediction_df)
            # print(f"Train Loss CE/Writer:{train_loss_ce}")
            # print(f"Train Writer Accuracy: {train_writer_acc}")
            print(f"Train Loss:{train_loss}")
            print(f"Train Loss CTC:{train_loss_ctc}")
            print(f"Train Loss CE/Writer:{train_loss_ce}")
            print(f"Train Loss Content: {train_loss_c}")
            print(f"Train Loss Style: {train_loss_s}")
            print(f"Train CER: {train_cer}")
            print(f"Train WER: {train_wer}")
            print(f"Train Writer Accuracy: {train_writer_acc}")
            print(f"Train Loss CTC out: {train_loss_ctc_out}")
            print(f"Train Loss CE out: {train_loss_ce_out}")
            print(f"Train Loss Autoencoder: {train_loss_auto}")
            print(f"Train Out CER: {train_out_cer}")
            print(f"Train Out WER: {train_out_wer}")
            print(f"Train Out Writer Accuracy: {train_out_writer_acc}")

            self.writer.add_scalars('Total Loss', {'train_loss': train_loss}, epoch)
            self.writer.add_scalars('CTC Loss', {'train_loss_ctc': train_loss_ctc}, epoch)
            self.writer.add_scalars('CE Loss', {'train_loss_ce': train_loss_ce}, epoch)
            self.writer.add_scalars('Content Loss', {'train_loss_c': train_loss_c}, epoch)
            self.writer.add_scalars('Style Loss', {'train_loss_s': train_loss_s}, epoch)
            self.writer.add_scalars('CER', {'train_cer': train_cer}, epoch)
            self.writer.add_scalars('WER', {'train_wer': train_wer}, epoch)
            self.writer.add_scalars('Writer', {'train_writer_acc': train_writer_acc}, epoch)
            self.writer.add_scalars('Auto Loss', {'train_loss_auto': train_loss_auto}, epoch)
            self.writer.add_scalars('Out CTC Loss', {'train_loss_ctc_out': train_loss_ctc_out}, epoch)
            self.writer.add_scalars('Out CE Loss', {'train_loss_ce_out': train_loss_ce_out}, epoch)
            self.writer.add_scalars('Out CER', {'train_out_cer': train_out_cer}, epoch)
            self.writer.add_scalars('Out WER', {'train_out_wer': train_out_wer}, epoch)
            self.writer.add_scalars('Out Writer', {'train_out_writer_acc': train_out_writer_acc}, epoch)

            # if epoch%100 == 0:
            #     self.get_plot()
            #     self.get_plot("val")
            #     self.save_model()

            # if epoch % self.config.model_eval_epoch == 0:
            if epoch % 1000 == 0:
                torch.cuda.empty_cache()

                eval_return_dict = self.eval_model()
                val_loss = eval_return_dict["total_loss"]
                val_loss_ctc = eval_return_dict["total_loss_ctc"]
                val_loss_ce = eval_return_dict["total_loss_ce"]
                val_loss_c = eval_return_dict["total_loss_c"]
                val_loss_s = eval_return_dict["total_loss_s"]
                val_cer = eval_return_dict["cer"]
                val_wer = eval_return_dict["wer"]
                val_writer_acc = eval_return_dict["writer_acc"]
                val_loss_ctc_out = eval_return_dict["total_loss_ctc_out"]
                val_loss_ce_out = eval_return_dict["total_loss_ce_out"]
                val_loss_auto = eval_return_dict["total_loss_auto"]
                val_out_cer = eval_return_dict["out_cer"]
                val_out_wer = eval_return_dict["out_wer"]
                val_out_writer_acc = eval_return_dict["out_writer_acc"]


                self.writer.add_scalars('Total Loss', {'val_loss': val_loss}, epoch)
                self.writer.add_scalars('CTC Loss', {'val_loss_ctc': val_loss_ctc}, epoch)
                self.writer.add_scalars('CE Loss', {'val_loss_ce': val_loss_ce}, epoch)
                self.writer.add_scalars('Content Loss', {'val_loss_c': val_loss_c}, epoch)
                self.writer.add_scalars('Style Loss', {'val_loss_s': val_loss_s}, epoch)
                self.writer.add_scalars('CER', {'val_cer': val_cer}, epoch)
                self.writer.add_scalars('WER', {'val_wer': val_wer}, epoch)
                self.writer.add_scalars('Writer', {'val_writer_acc': val_writer_acc}, epoch)
                self.writer.add_scalars('Auto Loss', {'val_loss_auto': val_loss_auto}, epoch)
                self.writer.add_scalars('Out CTC Loss', {'val_loss_ctc_out': val_loss_ctc_out}, epoch)
                self.writer.add_scalars('Out CE Loss', {'val_loss_ce_out': val_loss_ce_out}, epoch)
                self.writer.add_scalars('Out CER', {'val_out_cer': val_out_cer}, epoch)
                self.writer.add_scalars('Out WER', {'val_out_wer': val_out_wer}, epoch)
                self.writer.add_scalars('Out Writer', {'val_out_writer_acc': val_out_writer_acc}, epoch)

                self.get_plot()
                self.get_plot("val")
                self.save_model()

    def eval_model(self, dataset='val'):
        print(f"Evaluation on {dataset} set")
        if dataset == 'val':
            dataloader = self.val_set

        self.model.eval()

        total_loss = []
        total_loss_ctc = []
        total_loss_ce = []
        total_loss_c = []
        total_loss_s = []
        total_loss_ctc_out = []
        total_loss_ce_out = []
        total_loss_auto = []
        total_char_seq_prediction = []
        total_writer_prediction = []
        total_out_char_seq_prediction = []
        total_out_writer_prediction = []

        with progressbar.ProgressBar(max_value=len(dataloader)) as bar:

            for index, batch_output in enumerate(dataloader):


                with torch.no_grad():

                    return_dict = self.model(batch_output)

                    predicted_char_seq = return_dict["outputs"]["predicted_char_seq"]
                    predicted_writers = return_dict["outputs"]["predicted_writers"]
                    out_predicted_char_seq = return_dict["outputs"]["out_predicted_char_seq"]
                    out_predicted_writers = return_dict["outputs"]["out_predicted_writers"]
                    out_img = return_dict["outputs"]["out_img"]

                    loss_ctc = return_dict["losses"]["loss_ctc"]
                    loss_ce = return_dict["losses"]["loss_ce"]
                    loss_c = return_dict["losses"]["loss_content"]
                    loss_s = return_dict["losses"]["loss_style"]
                    loss_ctc_out = return_dict["losses"]["out_loss_ctc"]
                    loss_ce_out = return_dict["losses"]["out_loss_ce"]
                    loss_auto = return_dict["losses"]["loss_auto"]

                    loss = loss_ctc + loss_ce + loss_c + loss_s + loss_ctc_out + loss_ce_out + loss_auto


                total_loss.append(loss.item())
                total_loss_ctc.append(loss_ctc.item())
                total_loss_ce.append(loss_ce.item())
                total_loss_c.append(loss_c.item())
                total_loss_s.append(loss_s.item())
                total_loss_ctc_out.append(loss_ctc_out.item())
                total_loss_ce_out.append(loss_ce_out.item())
                total_loss_auto.append(loss_auto.item())

                total_char_seq_prediction.extend(self.store_char_seq_prediction(predicted_char_seq, batch_output["img_txt_txt_tgt"]))
                total_out_char_seq_prediction.extend(self.store_char_seq_prediction(out_predicted_char_seq, batch_output["img_txt_txt_tgt"]))
                total_writer_prediction.extend(self.store_writers_prediction(predicted_writers, batch_output['writer_id']))
                total_out_writer_prediction.extend(self.store_writers_prediction(out_predicted_writers, batch_output['writer_id']))

                bar.update(index)


        total_loss = np.average(total_loss)
        total_loss_ctc = np.average(total_loss_ctc)
        total_loss_ce = np.average(total_loss_ce)
        total_loss_c = np.average(total_loss_c)
        total_loss_s = np.average(total_loss_s)
        total_loss_ctc_out = np.average(total_loss_ctc_out)
        total_loss_ce_out = np.average(total_loss_ce_out)
        total_loss_auto = np.average(total_loss_auto)

        char_seq_prediction_df = pd.DataFrame(total_char_seq_prediction, columns=["target", "predicted"])
        df = char_seq_prediction_df
        cer = char_error_rate(preds=df.predicted.values.tolist(), target=df.target.values.tolist())
        wer = word_error_rate(preds=df.predicted.values.tolist(), target=df.target.values.tolist())
        acc = (df["target"] == df["predicted"]).sum()/df.shape[0]

        writer_prediction_df = pd.DataFrame(total_writer_prediction, columns=["target", "predicted"])
        writer_acc = (writer_prediction_df["target"] == writer_prediction_df["predicted"]).sum()/writer_prediction_df.shape[0]
        
        out_char_seq_prediction_df = pd.DataFrame(total_out_char_seq_prediction, columns=["target", "predicted"])
        out_df = out_char_seq_prediction_df
        out_cer = char_error_rate(preds=out_df.predicted.values.tolist(), target=out_df.target.values.tolist())
        out_wer = word_error_rate(preds=out_df.predicted.values.tolist(), target=out_df.target.values.tolist())
        out_acc = (out_df["target"] == out_df["predicted"]).sum()/out_df.shape[0]

        out_writer_prediction_df = pd.DataFrame(total_out_writer_prediction, columns=["target", "predicted"])
        out_writer_acc = (out_writer_prediction_df["target"] == out_writer_prediction_df["predicted"]).sum()/out_writer_prediction_df.shape[0]

        # print(char_seq_prediction_df)
        # print(writer_prediction_df)
        print(f"{dataset} Loss:{total_loss}")
        print(f"{dataset} Loss CTC:{total_loss_ctc}")
        print(f"{dataset} Loss CE/Writer:{total_loss_ce}")
        print(f"{dataset} Loss Content: {total_loss_c}")
        print(f"{dataset} Loss Style: {total_loss_s}")
        print(f"{dataset} CER: {cer}")
        print(f"{dataset} WER: {wer}")
        print(f"{dataset} Writer Accuracy: {writer_acc}")
        print(f"{dataset} Loss CTC out: {total_loss_ctc_out}")
        print(f"{dataset} Loss CE out: {total_loss_ce_out}")
        print(f"{dataset} Loss Autoencoder: {total_loss_auto}")
        print(f"{dataset} Out CER: {out_cer}")
        print(f"{dataset} Out WER: {out_wer}")
        print(f"{dataset} Out Writer Accuracy: {out_writer_acc}")

        eval_return_dict = {
                "total_loss": total_loss,
                "total_loss_ctc": total_loss_ctc,
                "total_loss_ce": total_loss_ce,
                "total_loss_c": total_loss_c,
                "total_loss_s": total_loss_s,
                "cer": cer,
                "wer": wer,
                "writer_acc": writer_acc,
                "total_loss_ctc_out": total_loss_ctc_out,
                "total_loss_ce_out": total_loss_ce_out,
                "total_loss_auto": total_loss_auto,
                "out_cer": out_cer,
                "out_wer": out_wer,
                "out_writer_acc": out_writer_acc,
                }

        return eval_return_dict

    def gen_model(self):
        pass

    def test_model(self):
        pass




