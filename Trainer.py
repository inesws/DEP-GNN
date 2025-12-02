import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import  MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.model = model.cuda()
        if params.loss == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().cuda()
        else:
            self.criterion = BCEWithLogitsLoss().cuda()

        self.best_model_states = None

        DEP_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "DEP" in name:
                DEP_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                    print('DEP in Freeze mode')
                else:
                    param.requires_grad = True
                    print('DEP in Train mode')

            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr_scaler is not None: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr },
                    {'params': other_params, 'lr': self.params.lr * self.params.multi_lr_scaler}
                ], weight_decay=self.params.weight_decay)
                print('Setting Regression head lr = {}'.format(self.params.lr * self.params.multi_lr_scaler))
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr_scaler is not None:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr* self.params.multi_lr_scaler}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        #self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        #)

        self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR( self.optimizer, 
                                                                   step_size=30,  # decrease LR every 20 epochs
                                                                   gamma=0.5      # multiply LR by 0.5 at each step
                                                                   )
        # Option A: Cosine Annealing with Warm Restarts

        #self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
        #                                                                 T_0=50,      # Restart every 50 epochs
        #                                                                 T_mult=1,    # Keep same cycle length
        #                                                                 eta_min=1e-7 # Very low minimum
        #                                                                 )
        
        self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR( self.optimizer,
                                milestones=[30, 60, 90, 120, 150, 180],  # Decay at these epochs
                                gamma=0.5  # Halve the LR at each milestone
                                )
        
        self.optimizer_scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=200,
                                                                         power=0.9 ) # Gentle polynomial decay)
        
        #self.optimizer_scheduler  = CosineAnnealingWarmRestarts(self.optimizer,
        #                                                        T_0=25,      # First restart at epoch 25 (5 frozen + 25)
        #                                                        T_mult=2,eta_min=1e-7)
        
        print(self.model)

    def get_metrics_for_regression(self, data_loader, model):
        model.eval()

        truths = []
        preds = []
        losses = []
        for batch in tqdm(data_loader, mininterval=1):
            x, y, _ = batch
            x = x.cuda().float()
            y = y.cuda().float()
            eps = 1e-6
            y_log = torch.log(y + eps)

            pred = model(x)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()
            #preds += torch.exp(pred).cpu().squeeze().numpy().tolist()

            
            #loss = self.criterion(pred, y_log)
            loss = self.criterion(pred, y)
            losses.append(loss.data.cpu().numpy())

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        n_rmse = rmse/ np.std(truths)
        loss = np.mean(losses)
        return corrcoef, r2, rmse, n_rmse, loss

    def _set_backbone_trainable(self, trainable):
        """Helper method to set backbone parameters trainable or not
        Args:
            trainable (bool): If True, backbone will be trainable (requires_grad=True)
                            If False, backbone will be frozen (requires_grad=False)
        """
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = trainable

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        n_rmse = 0
        loss_best = 10000
        epochs_no_improve = 0

        # Validate frozen_epochs logic
        if not self.params.frozen and hasattr(self.params, 'frozen_epochs') and self.params.frozen_epochs > 0:
            print("Warning: frozen_epochs > 0 but params.frozen is False. Setting frozen_epochs to 0 as backbone should be trainable from start.")
            self.params.frozen_epochs = 0

        for epoch in range(self.params.epochs):
            
            # Check if we should start training the backbone
            if self.params.frozen and hasattr(self.params, 'frozen_epochs') and epoch == self.params.frozen_epochs and self.params.frozen_epochs > 0:
                print(f"Epoch {epoch}: Making backbone trainable")
                for name, param in self.model.named_parameters():
                    if "backbone" in name:
                        param.requires_grad = True
                
                # Get current learning rate from scheduler for regression head
                current_lr = self.optimizer_scheduler.get_last_lr()[0]
                
                # Recreate optimizer with unfrozen parameters
                backbone_params = []
                other_params = []
                for name, param in self.model.named_parameters():
                    if "backbone" in name:
                        backbone_params.append(param)
                    else:
                        other_params.append(param)
                
                print(f"Unfreezing backbone - Original LR: {self.params.lr:.6f}, Current LR: {current_lr:.6f}")
                
                # Define parameter groups with their respective learning rates
                param_groups = [
                    {'params': backbone_params, 'lr': self.params.lr },  # Fresh start with original/scaled LR for backbone
                    {'params': other_params, 'lr': current_lr}  # Keep current LR for regression head
                ]

                # Create optimizer based on type
                if self.params.optimizer == 'AdamW':
                    self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.params.weight_decay)
                else:
                    self.optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=self.params.weight_decay)
                
                # Reset scheduler with remaining epochs
                remaining_steps = (self.params.epochs - epoch) * self.data_length
                self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=remaining_steps, eta_min=1e-6
                )
                #self.optimizer_scheduler  = CosineAnnealingWarmRestarts(self.optimizer,
                                                                        T_0=20,      # First restart at epoch 25 (5 frozen + 25)
                                                                        T_mult=2,eta_min=1e-7)


            self.model.train()
            start_time = timer()
            losses = []
            truths = []
            preds = []
            losses = []
            for batch in tqdm(self.data_loader['train'], mininterval=10):
                x, y = batch[0], batch[1]
                self.optimizer.zero_grad()
                x = x.cuda().float()
                y = y.cuda().float()

                eps = 1e-6
                y_log = torch.log(y + eps)

                pred = self.model(x)
                
                truths += y.detach().cpu().squeeze().numpy().tolist()
                preds += pred.detach().cpu().squeeze().numpy().tolist()
                #preds += torch.exp(pred).detach().cpu().squeeze().numpy().tolist()
                
                loss = self.criterion(pred, y)
                #loss = self.criterion(pred, y_log)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            truths = np.array(truths)
            preds = np.array(preds)
            t_corrcoef = np.corrcoef(truths, preds)[0, 1]
            t_r2 = r2_score(truths, preds)
            t_rmse = mean_squared_error(truths, preds) ** 0.5
            t_n_rmse = t_rmse/ np.std(truths)

            with torch.no_grad():
                #t_corrcoef, t_r2, t_rmse, t_n_rmse, _ = self.get_metrics_for_regression(self.data_loader['train'], self.model)
                v_corrcoef, v_r2, v_rmse, v_n_rmse, v_loss = self.get_metrics_for_regression(self.data_loader['val'], self.model)

                print(
                    "Epoch {} : Training Loss: {:.5f}, norm_rmse: {:.10f}, rmse: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        t_n_rmse,
                        t_rmse,
                        t_corrcoef,
                        t_r2,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(
                    "Val Loss: {:.5f}, norm_rmse: {:.10f}, rmse: {:.5f}, corrcoef: {:.5f}, r2: {:.5f},  LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        v_loss,
                        v_n_rmse,
                        v_rmse,
                        v_corrcoef,
                        v_r2,
                        v_rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )                
                if v_loss < loss_best: # if v_n_rmse < n_rmse_best
                    print("Val loss decreasing....saving weights !! ")

                    best_r2_epoch = epoch + 1
                    corrcoef_best = v_corrcoef
                    r2_best = v_r2
                    rmse_best = v_rmse
                    n_rmse_best = v_n_rmse
                    loss_best = v_loss

                    self.best_model_states = copy.deepcopy(self.model.state_dict())

                    epochs_no_improve = 0

                    if not os.path.isdir(self.params.model_dir):
                        os.makedirs(self.params.model_dir)
                    model_path = self.params.model_dir + "/epoch{}_NormRMSE_{:.5f}.pth".format(best_r2_epoch, n_rmse_best)
                    torch.save(self.model.state_dict(), model_path)
                    print("model save in " + model_path)

                else: # No val improvement
                    epochs_no_improve += 1
                
                if epochs_no_improve >= self.params.patience: # No val improvement for more than x epochs
                    print(f"Early stopping at epoch {epoch}. Best Val Loss: {loss_best:.6f}, Best Val RMSE: {n_rmse_best:.10f} (epoch {best_r2_epoch})")
                    break

        self.model.load_state_dict(self.best_model_states)



        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse, n_rmse, _ = self.get_metrics_for_regression(self.data_loader['test'], self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: norm_rmse: {:.10f}, rmse: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}".format(
                    n_rmse,
                    rmse,
                    corrcoef,
                    r2,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_NormRMSE_{:.5f}.pth".format(best_r2_epoch, n_rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
