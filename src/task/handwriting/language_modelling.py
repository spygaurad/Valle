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

from src.encoding.PositionalEncoding import PositionalEncoding, LearnablePositionalEncoding
from src.model.handwriting.lm import ARLM
from src.task.task import Task

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LanguageModel(nn.Module):
    def __init__(self, char_model, config, device):
        super(LanguageModel, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        vocab_size = self.char_model.vocab_size
        char_embedding = nn.Embedding(vocab_size, config.char_embedding_dim)

        # pos_encoding_char_dim = LearnablePositionalEncoding(config.char_embedding_dim)
        pos_encoding = PositionalEncoding(config.char_embedding_dim)

        self.lm = DataParallel(ARLM(char_embedding, pos_encoding, config, vocab_size, device))


        print(f"Number of parameters: {count_parameters(self.lm)}")

    def forward(self, batch_output):

        batch_output.device = self.device

        txt2txt_dec_txt = self.lm(batch_output['txt_txt_tgt_in'],
                                  batch_output['txt_txt_tgt_in_mask'],
                                  batch_output['txt_txt_tgt_in_pad_mask'])

        return txt2txt_dec_txt

    def evaluate(self, batch_output, type):

        batch_output.device = self.device

        if type == "txt2txt":
            try:
                txt2txt_dec_txt = self.lm(batch_output['txt_txt_tgt_in'],
                                          batch_output['txt_txt_tgt_in_mask'],
                                          batch_output['txt_txt_tgt_in_pad_mask'])
                return txt2txt_dec_txt

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for txt2txt task.")

    def generate(self, batch_output, type):

        batch_output.device = self.device

        if type == "txt2txt":
            txt2txt_dec_txt = self.lm.module.generate(self.char_model.char2index["TSOS"],
                                                      self.char_model.char2index["TEOS"],
                                                      self.config.max_char_len)

            return txt2txt_dec_txt

    def step(self, tgt, i):
        return self.lm.module.step(tgt, i)


class LanguageModelTask(Task):

    def __init__(self, train_set, val_set, test_set, train_gen_set, val_gen_set, test_gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, val_gen_set, char_model, config, device, exp_track)

        self.train_gen_set = train_gen_set
        self.val_gen_set = val_gen_set
        self.test_gen_set = test_gen_set

        self.train_batch_size = self.config.batch_size
        self.val_batch_size = self.config.batch_size
        self.writer = SummaryWriter(self.config.tensorboard_path)

        print("Language Model Task")

        print(f"Train Batch Size: {self.train_batch_size}")
        print(f"Val Batch Size: {self.val_batch_size}")

        self.use_scheduler = False
        self.warmup_epochs = 0
        self.output_dump_len = 20
        self.prev_val_loss = np.inf

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def build_model(self):
        return LanguageModel(self.char_model, self.config, self.device)

    def get_scheduler(self):
        # scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 500, eta_min=0.0005/10.0)
        scheduler = None
        return scheduler

    def get_scaler(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision_training)
        # scaler = None
        return scaler

    def loss_function(self):
        # txt = nn.CrossEntropyLoss(ignore_index=self.char_model.char2index['PAD'], label_smoothing=0.1)
        txt = nn.CrossEntropyLoss(ignore_index=self.char_model.char2index['PAD'])

        return {'txt': txt,}

    def get_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=0.1)

    def calculate_perplexity(self):
        pass

    def calculate_wer(self, target, predicted, raw_prob=False, return_raw=False, ctc_mode=False):
        wer = []
        raw_texts = []

        with torch.no_grad():
            if raw_prob:
                predicted = torch.argmax(predicted, dim=-1)
            bs = target.shape[0]
            token = "TEOS"
            for i in range(bs):
                str_target = (self.char_model.indexes2characters(target[i].cpu().numpy()))
                # print(str_target)

                if token in str_target:
                    str_target_first_pad_index = str_target.index(token)
                else:
                    str_target_first_pad_index = len(str_target)

                str_target = "".join(str_target[:str_target_first_pad_index])

                str_predicted =  (self.char_model.indexes2characters(predicted[i].cpu().numpy(), ctc_mode))
                # print(str_predicted)
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

                raw_texts.append((str_target, str_predicted))

                wer.append(editdistance.eval(str_target, str_predicted)/(len(str_target)))

            if return_raw:
                return raw_texts

            non_zeros = np.count_nonzero(wer)
            total = len(wer)
            acc = (total - non_zeros)/total

            wer = np.average(wer)
            return wer, acc

    def train_model(self):

        print("*****************************************************************")
        self.gen_model()
        print("*****************************************************************")
        print("*****************************************************************")
        self.eval_model(save_model=False)
        print("*****************************************************************")
        
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

            total_txt2txt_loss = []
            train_raw_texts = []
            total_train_perplexity = []
            print("Mixed precision:", self.use_mixed_precision_training)



            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision_training):

                        txt2txt_dec_txt = self.model(batch_output)
                        txt2txt_loss = self.criterion['txt'](txt2txt_dec_txt.view(-1, self.char_model.vocab_size),
                                                             batch_output['txt_txt_tgt_out'].contiguous().view(-1))
                        loss = txt2txt_loss

                    # raw_text = self.calculate_wer(batch_output['img_txt_txt_tgt_out'], img2txt_dec_txt, raw_prob=True, return_raw=True)
                    # train_raw_texts.extend(raw_text)

                    # loss.backward()
                    # self.optimizer.step()

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                   
                    total_train_perplexity.append(torch.exp(loss).item())
                    total_txt2txt_loss.append(txt2txt_loss.item())

                    bar.update(index)

            if self.use_scheduler:
                self.scheduler.step()

            total_loss = total_txt2txt_loss

            train_loss = np.average(total_loss)
            train_txt2txt_loss = np.average(total_txt2txt_loss)
            train_perplexity = np.average(total_train_perplexity)
            
            
            print("train_txt2txt_loss", (train_txt2txt_loss))
            print("train_total_loss", (train_loss))
            print("train_bpc", (train_txt2txt_loss/np.log(2)))
            print("train_perplexity", (train_perplexity))
            self.writer.add_scalars('Loss', {'train_loss': train_loss}, epoch)
            self.writer.add_scalars('Txt2Txt Loss', {'train_txt2txt_loss': train_txt2txt_loss}, epoch)
            self.writer.add_scalars('Txt2Txt Perplexity', {'train_perplexity': train_perplexity}, epoch)

            if train_raw_texts:
                train_df = pd.DataFrame(train_raw_texts, columns=["target", "predicted"])
                train_cer = char_error_rate(preds=train_df.predicted.values.tolist(), target=train_df.target.values.tolist())
                train_wer = word_error_rate(preds=train_df.predicted.values.tolist(), target=train_df.target.values.tolist())
                train_acc = (train_df["target"] == train_df["predicted"]).sum()/train_df.shape[0]
                print("train_cer", (train_cer))
                print("train_wer", (train_wer))
                print("train_acc", (train_acc))
                self.compute_modified_wer_cer(train_df.predicted.values.tolist(), train_df.target.values.tolist())

            if np.isnan(train_loss):
                print("Traning halted due to nan loss")
                break
            
            self.writer.add_scalars('Loss', {'train_loss': train_loss}, epoch)
            self.writer.add_scalars('Txt2Txt Loss', {'train_txt2txt_loss': train_txt2txt_loss}, epoch)

            if epoch % self.config.model_eval_epoch == 0:
                torch.cuda.empty_cache()

                self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
                self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

                if train_raw_texts:
                    train_df.to_csv(f"{self.config.gen_epoch_path}/train.csv", index=False)

                if self.exp_track is not None:

                    self.exp_track.log_metric("train_img2img_loss", np.average(total_img2img_loss))
                    self.exp_track.log_metric("train_img2txt_loss", np.average(total_img2txt_loss))
                    self.exp_track.log_metric("train_txt2img_loss", np.average(total_txt2img_loss))
                    self.exp_track.log_metric("train_total_loss", np.average(total_loss))

                val_loss, val_txt2txt_loss, val_perplexity = self.eval_model()
                self.writer.add_scalars('Loss', {'val_loss': val_loss}, epoch)
                self.writer.add_scalars('Txt2Txt Loss', {'val_txt2txt_loss': val_txt2txt_loss}, epoch)
                self.writer.add_scalars('Txt2Txt Perplexity', {'val_perplexity': val_perplexity}, epoch)

                self.save_model()
                self.gen_model()

    @staticmethod
    def compute_modified_wer_cer(predicted, references):
        def format_string_for_wer(str):
            str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
            str = re.sub('([ \n])+', " ", str).strip()
            return str


        def edit_wer_from_list(truth, pred):
            edit = 0
            for pred, gt in zip(pred, truth):
                gt = format_string_for_wer(gt)
                pred = format_string_for_wer(pred)
                gt = gt.split(" ")
                pred = pred.split(" ")
                edit += editdistance.eval(gt, pred)
            return edit

        def nb_words_from_list(list_gt):
            len_ = 0
            for gt in list_gt:
                gt = format_string_for_wer(gt)
                gt = gt.split(" ")
                len_ += len(gt)
            return len_


        def nb_chars_from_list(list_gt):
            return sum([len(t) for t in list_gt])

        predicted = [re.sub("( )+", ' ', t).strip(" ") for t in predicted]
        cer_wo_norm = [editdistance.eval(u, v) for u,v in zip(predicted, references)]
        cer_norm =  nb_chars_from_list(references)
        cer = sum(cer_wo_norm)/cer_norm

        wer_wo_norm = edit_wer_from_list(predicted, references)
        wer_norm = nb_words_from_list(references)
        wer = wer_wo_norm/wer_norm

        print("CER Updated:", cer)
        print("WER Updated:", wer)

    def compute_greedy_wer2(self, dataset, mode="val", search_type="greedy", beam_size=None):
        raw_texts = []
        self.model.eval()
        print(f"Search type: {search_type}")
        with torch.no_grad():
            with progressbar.ProgressBar(max_value=len(dataset)) as bar:
                for index, batch_output in enumerate(dataset):

                    if search_type == "greedy":
                        output = self.model.generate(batch_output, "img2txt")
                    else:
                        output, score = self.model.beam_search(batch_output, "img2txt", beam_size)

                    output = output[1:]
                    output = torch.tensor(output).unsqueeze(0)
                    raw_text = self.calculate_wer(batch_output['img_txt_txt_tgt_out'], output, raw_prob=False, return_raw=True)
                    raw_texts.extend(raw_text)

                    bar.update(index)

                df = pd.DataFrame(raw_texts, columns=["target", "predicted"])

                self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
                self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

                df.to_csv(f"{self.config.gen_epoch_path}/{mode}.csv", index=False)

                greedy_cer = char_error_rate(preds=df.predicted.values.tolist(), target=df.target.values.tolist())
                greedy_wer = word_error_rate(preds=df.predicted.values.tolist(), target=df.target.values.tolist())
                greedy_acc = (df["target"] == df["predicted"]).sum()/df.shape[0]

                print(f"Greedy {mode} cer:", np.average(greedy_cer))
                print(f"Greedy {mode} wer:", np.average(greedy_wer))
                print(f"Greedy {mode} acc:", np.average(greedy_acc))

                self.compute_modified_wer_cer(df.predicted.values.tolist(), df.target.values.tolist())

                return greedy_cer

    def eval_model(self, one_by_one=False, index=None, save_model=True, dataset=None):

        if dataset is None:
            dataset=self.val_set

        self.model.eval()

        total_txt2txt_loss = []
        val_encoder_raw_texts = []
        total_val_perplexity = []

        with torch.no_grad():

            print("\nEvaluating..............")

            with progressbar.ProgressBar(max_value=len(dataset)) as bar:

                for index, batch_output in enumerate(dataset):

                    try:
                        txt2txt_dec_txt = self.model.evaluate(batch_output, "txt2txt")
                        txt2txt_loss = self.criterion['txt'](txt2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                             batch_output['txt_txt_tgt_out'].contiguous().view(-1))
                        total_txt2txt_loss.append(txt2txt_loss.item())
                        total_val_perplexity.append(torch.exp(txt2txt_loss).item())
                    except RuntimeWarning:
                        pass

                    bar.update(index)

            total_loss =  total_txt2txt_loss


        val_loss = np.average(total_loss)
        val_txt2txt_loss = np.average(total_txt2txt_loss)
        val_perplexity = np.average(total_val_perplexity)
        print("val_total_loss", (val_loss))
        print("val_txt2txt_loss", (val_txt2txt_loss))
        print("val_bpc", (val_txt2txt_loss/np.log(2)))
        print("val_perplexity", (val_perplexity))

        return val_loss, val_txt2txt_loss, val_perplexity


    def gen_model(self):

        def save_txt(file_path, data, txt_title):
            txt = "".join(data)
            f = open(file_path, "w")
            f.write("".join(txt))
            f.close()

            if self.exp_track is not None:
                self.exp_track.log_text(txt_title, txt)

        def save_img(img_path, img, img_title):
            cv2.imwrite(img_path, img)

            if self.exp_track is not None:

                self.exp_track.log_image(img_title, img.squeeze())

        def gen_output(gen_set, mode="test"):

            with torch.no_grad():

                counter = {"label_1":0, "label_2":0, "label_0":0}
                test_dump_len = self.output_dump_len

                for index, batch_output in enumerate(gen_set, 1):

                    if batch_output['label'] == 0 and counter["label_0"] < test_dump_len:

                        output = self.model.generate(batch_output, "txt2txt")

                        real_output = self.char_model.indexes2characters(batch_output['txt_txt_tgt_out'].cpu().numpy()[0])
                        save_txt(f"{self.config.gen_epoch_path}/{mode}_txt2txt_in_{index}.txt", real_output, 'real_txt2txt')

                        predicted_output = self.char_model.indexes2characters(output[1:])
                        save_txt(f"{self.config.gen_epoch_path}/{mode}_txt2txt_out_{index}.txt",
                                 predicted_output, 'predicted_txt2txt')
                        counter["label_0"] += 1

                    if counter["label_1"] >= test_dump_len:
                        break


        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        gen_output(self.train_gen_set, mode="train")
        gen_output(self.val_gen_set, mode="val")
        gen_output(self.test_set, mode="test")

    def test_model(self):
        pass
