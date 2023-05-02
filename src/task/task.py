import os
from abc import ABC, abstractmethod
from collections import defaultdict

import glob
import torch
import torch.nn as nn


class Task(ABC):

    def __init__(self, train_set, val_set, test_set, gen_set, char_model, config, device, exp_track):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.gen_set = gen_set
        self.char_model = char_model
        self.config = config
        self.device = device
        self.exp_track = exp_track
        self.use_mixed_precision_training = config.mixed_precision_training

        self.current_epoch = 1

        self.model = self.build_model()


        self.model = self.model.to(self.device)

        self.criterion = self.loss_function()
        self.optimizer = self.get_optimizer()
        
        self.scheduler = self.get_scheduler()

        self.scaler = self.get_scaler()

        self.results = defaultdict(list)

        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.load_model()
        # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)
        
        self.model = nn.parallel.DataParallel(self.model)

        # self.model = torch.compile(self.model, mode="default", backend="inductor")
        # self.model = torch.compile(self.model)
        # self.model = (self.model, device_ids=[self.device], find_unused_parameters=True)


    @abstractmethod
    def get_scaler(self):
        pass

    @abstractmethod
    def get_scheduler(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def loss_function(self):
        pass

    @abstractmethod
    def get_optimizer(self):
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load_model(self, copy_weights=False):
        # for name, param in model_dest.named_parameters():
        #         if name in model_src:
        #                     param.data.copy_(model_src[name].data)
        if self.config.run_id and self.config.model_id:
            self.config.checkpoint_path = self.config.EXP_DIR/self.config.run_id/'model'
            model_path = self.config.checkpoint_path/f"{self.config.model_id}.pth"
            # checkpoint = torch.load(model_path, map_location=f"cuda:{self.device}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model

            self.results = checkpoint["results"]
            self.scaler = checkpoint["scaler"]


            self.current_epoch = checkpoint["epoch"] + 1

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
                # self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                # for index,state_dict in enumerate(checkpoint["scheduler_state_dict"]):
                #     self.scheduler[index].load_state_dict(state_dict)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded model:{model_path}")

    def save_model(self, best=False):
        self.config.model_id = self.current_epoch

        model_name = f"{self.config.model_id}.pth"
        if best:
            checkpoints = glob.glob(os.path.join(str(self.config.best_checkpoint_path),"*.pth"))
            model_path = self.config.best_checkpoint_path/model_name
        else:
            checkpoints = glob.glob(os.path.join(str(self.config.checkpoint_path),"*.pth"))
            model_path = self.config.checkpoint_path/model_name

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            "epoch":  self.current_epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # "scheduler_state_dict": self.scheduler.state_dict(),
            # "scheduler_state_dict": [i.state_dict() for i in self.scheduler],
            "results": self.results,
            "scaler": self.scaler,
        }, model_path)

        # torch.save(model_to_save, self.config.checkpoint_path/f"model_only_{self.config.model_id}.pth")

        checkpoints = sorted(checkpoints, key=os.path.getmtime)
        if len(checkpoints) > 5:
            os.remove(checkpoints[0])

        print(f"\nSaved model:{model_path}\n")

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def eval_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass

    @abstractmethod
    def gen_model(self):
        pass
