import editdistance
import itertools
import numpy as np
import pandas as pd
import progressbar
import random
import re
import torch 
from torchmetrics.functional import char_error_rate, word_error_rate

from src.config.handwriting_config import HandwritingConfig
from src.data.handwriting.handwriting_dataloader import HandwritingDataloader
from src.task.handwriting.txt2img_with_style import Handwriting_Model
from src.task.handwriting.language_modelling import LanguageModel

# torch.set_num_threads(4)

random.seed(27)
np.random.seed(27)
torch.manual_seed(27)
torch.cuda.manual_seed_all(27)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_descriptor')

def calculate_wer_from_bach_tensor(char_model, target, predicted, raw_prob=False, return_raw=False, ctc_mode=False):
    wer = []
    raw_texts = []

    with torch.no_grad():
        if raw_prob:
            predicted = torch.argmax(predicted, dim=-1)
        bs = target.shape[0]
        token = "TEOS"
        for i in range(bs):
            str_target = (char_model.indexes2characters(target[i].cpu().numpy()))

            if token in str_target:
                str_target_first_pad_index = str_target.index(token)
            else:
                str_target_first_pad_index = len(str_target)

            str_target = "".join(str_target[:str_target_first_pad_index])

            str_predicted =  (char_model.indexes2characters(predicted[i].cpu().numpy(), ctc_mode))
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

def calculate_wer_dataset_parallel(args):

    datapoint, model, char_model, config, mode, search_type, beam_size, lm_model = args
    model.eval()

    beam_penalty = 0.6
    lm_weight =  0.05

    # print(f"Search type: {search_type}")
    # if search_type == "beam":
    #     print(f"Beam Size: {beam_size}")
    # if lm_model:
    #     print(f"Using Language Model in decoding")
    #     lm_model.eval()

    #     print(f"Beam Penalty: {beam_penalty}")
    #     print(f"LM Weight: {lm_weight}")

    with torch.no_grad():
        output, score = model.beam_search(datapoint, "img2txt", beam_size, lm_model=lm_model, beam_penalty=beam_penalty, lm_weight=lm_weight, char_model=char_model)
        output = output.unsqueeze(0)
        output = output[:, 1:]
        raw_text = calculate_wer_from_bach_tensor(char_model, datapoint['img_txt_txt_tgt_out'], output, raw_prob=False, return_raw=True)
        # print(raw_text)
        return raw_text[0]

def calculate_wer_dataset(model, char_model, config, dataset, mode="val", search_type="greedy", beam_size=None, lm_model=None, **kwargs):
    raw_texts = []
    model.eval()

    beam_penalty = kwargs.pop("beam_penalty", 0.5)
    lm_weight = kwargs.pop("lm_weight", 0.1)

    print(f"Search type: {search_type}")
    if search_type == "beam":
        print(f"Beam Size: {beam_size}")
    if lm_model:
        print(f"Using Language Model in decoding")
        lm_model.eval()

        print(f"Beam Penalty: {beam_penalty}")
        print(f"LM Weight: {lm_weight}")

    with torch.no_grad():
        with progressbar.ProgressBar(max_value=len(dataset)) as bar:
            for index, batch_output in enumerate(dataset):

                if search_type == "greedy":
                    output = model.generate(batch_output, "img2txt")
                else:
                    output, score = model.beam_search(batch_output, "img2txt", beam_size, lm_model=lm_model, beam_penalty=beam_penalty, lm_weight=lm_weight, char_model=char_model)
                    output = output.unsqueeze(0)

                # output = output[1:]
                # output = torch.tensor(output).unsqueeze(0)
                output = output[:, 1:]
                raw_text = calculate_wer_from_bach_tensor(char_model, batch_output['img_txt_txt_tgt_out'], output, raw_prob=False, return_raw=True)
                raw_texts.extend(raw_text)

                bar.update(index)

            df = pd.DataFrame(raw_texts, columns=["target", "predicted"])

            # config.gen_epoch_path = config.gen_path/"shallow_fusion"
            # config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

            # df.to_csv(f"{config.gen_epoch_path}/{mode}.csv", index=False)

            greedy_cer = char_error_rate(preds=df.predicted.values.tolist(), target=df.target.values.tolist())
            greedy_wer = word_error_rate(preds=df.predicted.values.tolist(), target=df.target.values.tolist())
            greedy_acc = (df["target"] == df["predicted"]).sum()/df.shape[0]

            print(f"Greedy {mode} cer:", np.average(greedy_cer))
            print(f"Greedy {mode} wer:", np.average(greedy_wer))
            print(f"Greedy {mode} acc:", np.average(greedy_acc))

            compute_modified_wer_cer(df.predicted.values.tolist(), df.target.values.tolist())

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


def load_model(model, device, config, copy_weights=False):
    # for name, param in model_dest.named_parameters():
    #         if name in model_src:
    #                     param.data.copy_(model_src[name].data)
    if config.run_id and config.model_id:
        config.checkpoint_path = config.EXP_DIR/config.run_id/'model'
        model_path = config.checkpoint_path/f"{config.model_id}.pth"
        checkpoint = torch.load(model_path, map_location=device)
        model_to_save = model.module if hasattr(model, "module") else model

        results = checkpoint["results"]
        scaler = checkpoint["scaler"]


        current_epoch = checkpoint["epoch"] + 1

        # print(checkpoint["optimizer_state_dict"])

        if copy_weights:
            # print(checkpoint["model_state_dict"])
            model_dest = model_to_save
            model_src = checkpoint["model_state_dict"]
            for name, param in model_dest.named_parameters():
                if name in model_src:
                    param.data.copy_(model_src[name].data)
        else:
            model_to_save.load_state_dict(checkpoint["model_state_dict"])
            # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded model:{model_path}")


def main():
    pass



if __name__=="__main__":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        handwriting_config = HandwritingConfig.from_json_file('src/config/json_file/handwriting.json')
        handwriting_config.run_id = "test"
        handwriting_config.model_id = "4460"
        handwriting_config.model_id = "6380"

        train_set, val_set, test_set, train_gen_set, val_gen_set, test_gen_set, char_model, _ = HandwritingDataloader(handwriting_config)("full")

        handwriting_model = Handwriting_Model(char_model, handwriting_config, device)
        load_model(handwriting_model, device, handwriting_config)
        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "greedy")
        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", 1)
        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", 8)

        # lm_config = HandwritingConfig.from_json_file('src/config/json_file/handwriting.json')
        # lm_config.run_id = "test"
        # lm_config.model_id = "16000"
        # lm_model = LanguageModel(char_model, lm_config, device)
        # load_model(lm_model, device, lm_config)
        from transformers import ReformerModelWithLMHead
        lm_model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8").to(device)
        for param in lm_model.parameters():
            param.requires_grad = False

        # beam_size = [4, 8, 16, 32, 64]
        beam_size = [32, ]
        beam_penalty = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        lm_weight = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]

        for bs, bp, lm in itertools.product(beam_size, beam_penalty, lm_weight):
            print("****************************************************************************************************************************************")
            print("Eval Set:")
            calculate_wer_dataset(handwriting_model, char_model, handwriting_config, val_gen_set, "val", "beam", beam_size=bs, lm_model=lm_model, beam_penalty=bp, lm_weight=lm)
            print("Test Set:")
            calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", beam_size=bs, lm_model=lm_model, beam_penalty=bp, lm_weight=lm)
            print("****************************************************************************************************************************************")


        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", beam_size=8, lm_model=lm_model, beam_penalty=0.6, lm_weight=0.05)
        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, val_gen_set, "val", "beam", beam_size=8, lm_model=lm_model, beam_penalty=0.6, lm_weight=0.05)
        """

        raw_text = []
        args = []
        mode = "test"
        
        for index, batch_output in enumerate(test_gen_set):
            args.append((batch_output, handwriting_model, char_model, handwriting_config, "test", "beam", 8, lm_model))

        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')
        mp.set_start_method('spawn')
        # mp.spawn(calculate_wer_dataset_parallel, args=args, nprocs=4, join=True)
        with mp.Pool(processes=4) as pool:
            # raw_texts = pool.map(calculate_wer_dataset_parallel, args[:10], chunksize=None)
            import tqdm
            raw_texts = list(tqdm.tqdm(pool.imap_unordered(calculate_wer_dataset_parallel, args, chunksize=10), total=len(test_gen_set)))
            print(raw_texts)

            df = pd.DataFrame(raw_texts, columns=["target", "predicted"])
            greedy_cer = char_error_rate(preds=df.predicted.values.tolist(), target=df.target.values.tolist())
            greedy_wer = word_error_rate(preds=df.predicted.values.tolist(), target=df.target.values.tolist())
            greedy_acc = (df["target"] == df["predicted"]).sum()/df.shape[0]

            print(f"Greedy {mode} cer:", np.average(greedy_cer))
            print(f"Greedy {mode} wer:", np.average(greedy_wer))
            print(f"Greedy {mode} acc:", np.average(greedy_acc))

            compute_modified_wer_cer(df.predicted.values.tolist(), df.target.values.tolist())
        # for index, batch_output in enumerate(test_gen_set):
        #     out = calculate_wer_dataset_parallel(batch_output, handwriting_model, char_model, handwriting_config, "test", "beam", beam_size=8, lm_model=lm_model, beam_penalty=0.6, lm_weight=0.05)
        #     print(out)
        """


        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", 8, lm_model=lm_model)
        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", 1)
        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", 1, lm_model=lm_model)
        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", 8)
        # calculate_wer_dataset(handwriting_model, char_model, handwriting_config, test_gen_set, "test", "beam", 8, lm_model=lm_model)


