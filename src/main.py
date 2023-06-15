import argparse
import time
import random

import neptune.new as neptune
import numpy as np
import torch
import torch.multiprocessing

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from src.config.handwriting_config import HandwritingConfig
from src.data.handwriting.handwriting_dataloader import HandwritingDataloader
from src.task.handwriting.handwriting import Handwriting
from src.task.handwriting.img2img import Img2Img as Handwriting_Img2Img
from src.task.handwriting.txt2txt import Txt2Txt as Handwriting_Txt2Txt
from src.task.handwriting.txt2img import Txt2Img as Handwriting_Txt2Img
from src.task.handwriting.img2txt import Img2Txt as Handwriting_Img2Txt
from src.task.handwriting.imgtxt_enc import ImgTxt_Enc as Handwriting_ImgTxt_Enc
from src.task.handwriting.img2txt_txt2img import Img2Txt_Txt2Img as Handwriting_Img2Txt_Txt2Img
from src.task.handwriting.img2img_txt2img import Img2Img_Txt2Img as Handwriting_Img2Img_Txt2Img
from src.task.handwriting.txt2img_img2img_imgtxt_enc import Txt2Img_Img2Img_ImgTxt_Enc
from src.task.handwriting.img2img_txt2img_img2txt import Handwriting as Handwriting_Img2Img_Txt2Img_Img2Txt
from src.task.handwriting.txt2img_with_style import Handwriting as Handwriting_Txt2Img_With_Style
from src.task.handwriting.style_transfer import StyleTransfer
from src.task.handwriting.language_modelling import LanguageModelTask

# torch.set_float32_matmul_precision('high')
# import warnings
# warnings.filterwarnings("ignore")
# torch.set_num_threads(4)

random.seed(27)
np.random.seed(27)
torch.manual_seed(27)
torch.cuda.manual_seed_all(27)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.multiprocessing.set_sharing_strategy('file_descriptor')

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def setup_out_dir(config):

    if not config.run_id:
        config.run_id = str(int(time.time()))
    config.checkpoint_path = config.EXP_DIR/config.run_id/'model'
    config.checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Model Checkpoint Path: {config.checkpoint_path}")

    config.best_checkpoint_path = config.EXP_DIR/config.run_id/'best_model'
    config.best_checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Model Checkpoint Path: {config.best_checkpoint_path}")

    config.gen_path = config.EXP_DIR/config.run_id/'gen_result'
    config.gen_path.mkdir(parents=True, exist_ok=True)

    config.tensorboard_path = config.EXP_DIR/config.run_id/'tensorboard'
    config.tensorboard_path.mkdir(parents=True, exist_ok=True)


def main(args):
    # dist.init_process_group(backend='nccl')



    if args.task in ['handwriting',
                     'handwriting-img2img',
                     'handwriting-txt2txt',
                     'handwriting-txt2img',
                     'handwriting-img2txt',
                     'handwriting-imgtxt_enc',
                     'handwriting-img2txt_txt2img',
                     'handwriting-img2img_txt2img',
                     'handwriting-txt2img_img2img_imgtxt_enc',
                     'handwriting-img2img_txt2img_img2txt',
                     'handwriting-txt2img_with_style',
                     'handwriting-lm',
                     'style-transfer']:
        config = HandwritingConfig.from_json_file('src/config/json_file/handwriting.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = torch.cuda.device_count()
    # config.update_batch_size(device_count)
    # device = int(os.environ['LOCAL_RANK'])
    # device = 0

    print("Detected Device:", device)
    print("Let's use", device_count, "GPUs!\n")

    # dist.barrier()
    # setup_for_distributed(device == 0)
    try:
        train_set, val_set, test_set, train_gen_set, val_gen_set, test_gen_set, char_model, _ = HandwritingDataloader(config)(args.data_type)
    except Exception as e:
        print('Error in Handwriting Dataloader: ', e)

    config.run_id = args.run_id
    config.model_id = args.model_id
    setup_out_dir(config)

    exp_param = dict(config.__dict__)
    exp_param['checkpoint_path'] = str(exp_param['checkpoint_path'])
    exp_param['gen_path'] = str(exp_param['gen_path'])

    try:
        if args.neptune_exp_id:
            exp_track = neptune.init(project=args.neptune_project_name, run=args.neptune_exp_id)
        else:
            exp_track = neptune.init(project=args.neptune_project_name)

        exp_track["parameters"] = exp_param
    except Exception:
        print("Neptune is not being used")
        exp_track = None

    if args.task == 'handwriting':
        task = Handwriting(train_set, val_set, test_set, gen_set,
                           char_model, config, device, exp_track)

    elif args.task == 'handwriting-img2img':
        task = Handwriting_Img2Img(train_set, val_set, test_set, gen_set,
                                   char_model, config, device, exp_track)

    elif args.task == 'handwriting-txt2txt':
        task = Handwriting_Txt2Txt(train_set, val_set, test_set, gen_set,
                                   char_model, config, device, exp_track)

    elif args.task == 'handwriting-txt2img':
        task = Handwriting_Txt2Img(train_set, val_set, test_set, gen_set,
                                   char_model, config, device, exp_track)

    elif args.task == 'handwriting-img2img':
        task = Handwriting_Img2Img(train_set, val_set, test_set, gen_set,
                                   char_model, config, device, exp_track)

    elif args.task == 'handwriting-img2txt':
        task = Handwriting_Img2Txt(train_set, val_set, test_set, gen_set,
                                   char_model, config, device, exp_track)

    elif args.task == 'handwriting-imgtxt_enc':
        task = Handwriting_ImgTxt_Enc(train_set, val_set, test_set, gen_set,
                                      char_model, config, device, exp_track)

    elif args.task == 'handwriting-img2txt_txt2img':
        task = Handwriting_Img2Txt_Txt2Img(train_set, val_set, test_set, gen_set,
                                           char_model, config, device, exp_track)

    elif args.task == 'handwriting-img2img_txt2img':
        task = Handwriting_Img2Img_Txt2Img(train_set, val_set, test_set, gen_set,
                                           char_model, config, device, exp_track)

    elif args.task == 'handwriting-txt2img_img2img_imgtxt_enc':
        task = Txt2Img_Img2Img_ImgTxt_Enc(train_set, val_set, test_set, gen_set, char_model,
                                          config, device, exp_track)
    elif args.task == 'handwriting-img2img_txt2img_img2txt':
        task = Handwriting_Img2Img_Txt2Img_Img2Txt(train_set, val_set, test_set, train_gen_set, val_gen_set, char_model,
                                                    config, device, exp_track)
    elif args.task == 'handwriting-txt2img_with_style':
        task = Handwriting_Txt2Img_With_Style(train_set, val_set, test_set, train_gen_set, val_gen_set, test_gen_set, char_model,
                                                    config, device, exp_track)
    elif args.task == 'handwriting-lm':
        task = LanguageModelTask(train_set, val_set, test_set, train_gen_set, val_gen_set, test_gen_set, char_model,
                                                    config, device, exp_track)
    elif args.task == 'style-transfer':
        task = StyleTransfer(train_set, val_set, test_set, train_gen_set, val_gen_set, test_gen_set, char_model,
                                                    config, device, exp_track)

    if args.mode == "train":
        task.train_model()
    elif args.model == "gen":
        task.gen_model()
    
    destroy_process_group()


def get_args():

    args = argparse.ArgumentParser()

    args.add_argument('--mode',
                      required=False,
                      type=str,
                      default='train',
                      choices=['train', 'gen'],
                      help='Mode in which we have to run the model')

    args.add_argument('--task',
                      required=True,
                      type=str,
                      choices=['handwriting',
                               'handwriting-img2img',
                               'handwriting-txt2txt',
                               'handwriting-txt2img',
                               'handwriting-img2txt',
                               'handwriting-imgtxt_enc',
                               'handwriting-img2txt_txt2img',
                               'handwriting-img2img_txt2img',
                               'handwriting-txt2img_img2img_imgtxt_enc',
                               'handwriting-img2img_txt2img_img2txt',
                               'handwriting-txt2img_with_style',
                               'handwriting-lm',
                               'style-transfer'],
                      help='Task type')

    args.add_argument('--data_type',
                      required=False,
                      type=str,
                      default='full',
                      choices=['full', 'small'],
                      help='Whether to load full data or small data for testing purposes')

    args.add_argument('--run_id',
                      required=False,
                      type=str,
                      default=None,
                      help="Dedicated out folder for the current run")

    args.add_argument('--model_id',
                      required=False,
                      type=str,
                      default=None,
                      help="Model unique identifier")

    args.add_argument('--neptune_project_name',
                      required=True,
                      type=str,
                      help="Neptune Project Name in which we have to track experiment")

    args.add_argument('--neptune_exp_id',
                      required=False,
                      type=str,
                      default=None,
                      help="Neptune experiment id if we want to append the logs in previous experiment")

    return args.parse_args()


if __name__ == "__main__":
    device_count = torch.cuda.device_count()
    print("Let's use", device_count, "GPUs!\n")

    args = get_args()

    main(args)
