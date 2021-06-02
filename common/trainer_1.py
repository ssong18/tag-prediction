import tqdm
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from models.tag_prediction_models import TagPredictor
from sklearn.metrics import f1_score


class Trainer():
    def __init__(self,
                 args,
                 train_loader,
                 val_loader,
                 test_loader,
                 question_info_dict,
                 answerer_info_dict,
                 tag_to_idx,
                 device):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.question_info_dict = question_info_dict
        self.answerer_info_dict = answerer_info_dict
        self.tag_to_idx = tag_to_idx
        self.device = device

        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()
        self.optimizer, self.scheduler = self._initialize_optimizer()

    def _initialize_model(self):
        model = TagPredictor(self.args,
                             self.question_info_dict,
                             self.answerer_info_dict,
                             self.tag_to_idx,
                             self.device)
        return model.to(self.device)

    def _initialize_criterion(self):
        criterion = torch.nn.MultiMarginLoss()
        #criterion = torch.nn.CrossEntropyLoss()
        #criterion = torch.nn.BCELoss()
        return criterion.to(self.device)

    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.args.lr,
                                                        epochs=self.args.epochs,
                                                        steps_per_epoch=len(self.train_loader),
                                                        pct_start=0.2)
        return optimizer, scheduler

    def train(self):
        # validate the initial-parameter
        summary = self.validate()
        best_F1 = summary['F1']
        wandb.log(summary, 0)
        # m = torch.nn.Sigmoid()

        # start training
        losses = []
        for e in range(1, self.args.epochs+1):
            print('[Epoch:%d]' % e)
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                x_batch = [feature.to(self.device) for feature in x_batch]
                y_batch = y_batch.to(self.device)
                # In training, only index(0) is the positive label
                logits = self.model(x_batch).squeeze(1)
                    
                # logits.shape: (N, 1 + negative_samples)
                loss = self.criterion(logits, torch.argmax(y_batch, 1))
                #loss = self.criterion(m(logits), y_batch.to(torch.float))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                losses.append(loss.item())

            if e % self.args.evaluate_every == 0:
                summary = self.validate()
                summary['loss'] = np.mean(losses)
                if summary['F1'] > best_F1:
                    best_F1 = summary['F1']
                wandb.log(summary, e)
                losses = []

    def validate(self):
        summary = {}
        y_pred_list = []
        y_true_list = []
        for x_batch, y_batch in tqdm.tqdm(self.val_loader):
            x_batch = [feature.to(self.device) for feature in x_batch]
            logits = self.model(x_batch).squeeze(1)
            y_pred = torch.exp(F.log_softmax(logits, 1)).detach().cpu().numpy()
            y_true = y_batch.cpu().numpy()

            y_pred_list.append(y_pred)
            y_true_list.append(y_true)

        y_pred_list = np.concatenate(y_pred_list)  # [0.4, 0.2, 0.1, 0.1, 0.2]
        y_true_list = np.concatenate(y_true_list)  # [  1,   0,   0,   0,   1]
        thresholds = np.arange(0, 1, 0.1)
        best_f1 = 0
        for th in thresholds:
            f1 = f1_score(y_true_list, y_pred_list > th, average='micro')
            if f1 > best_f1:
                best_f1 = f1
        summary['F1'] = best_f1

        return summary

    def test(self):
        pass