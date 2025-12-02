import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class TrainerGNN(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.model = model.cuda()

        if params.loss == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().cuda()
        else:
            self.criterion = BCEWithLogitsLoss().cuda()

        self.best_model_states = None

        # Get model parameters
        model_params = list(self.model.parameters())
        
        # Create optimizer
        if self.params.optimizer == 'Adam':
            if params.weight_decay is not None:
                self.optimizer = torch.optim.AdamW(model_params, lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
            else: 
                self.optimizer = torch.optim.AdamW(model_params, lr=self.params.lr)
        
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


    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        n_rmse = 0
        loss_best = 10000
        epochs_no_improve = 0

        for epoch in range(self.params.epochs):
            
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

                pred = self.model(x)
                
                truths += y.detach().cpu().squeeze().numpy().tolist()
                preds += pred.detach().cpu().squeeze().numpy().tolist()
                
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()

            optim_state = self.optimizer.state_dict()

            truths = np.array(truths)
            preds = np.array(preds)
            
            # Regression metrics
            truths = np.array(truths)
            preds = np.array(preds)
            t_corrcoef = np.corrcoef(truths, preds)[0, 1]
            t_r2 = r2_score(truths, preds)
            t_rmse = mean_squared_error(truths, preds) ** 0.5
            t_n_rmse = t_rmse/ np.std(truths)

            with torch.no_grad():
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
                    "Val Loss: {:.5f}, norm_rmse: {:.10f}, rmse: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        v_loss,
                        v_n_rmse,
                        v_rmse,
                        v_corrcoef,
                        v_r2,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )                
                if v_loss < loss_best:
                    print("Val loss decreasing....saving weights !! ")

                    best_epoch = epoch + 1
                    loss_best = v_loss

                    self.best_model_states = copy.deepcopy(self.model.state_dict())

                    epochs_no_improve = 0

                    if not os.path.isdir(self.params.model_dir):
                        os.makedirs(self.params.model_dir)

                    # Save model
                    model_path = self.params.model_dir + "/model_epoch{}_loss_{:.5f}.pth".format(best_epoch, loss_best)
                    torch.save(self.model.state_dict(), model_path)
                    print(f"Model saved in {model_path}")

                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= self.params.patience:
                    print(f"Early stopping at epoch {epoch}. Best Val Loss: {loss_best:.6f} (epoch {best_epoch})")
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
            model_path = self.params.model_dir + "/final_model_NormRMSE_{:.5f}.pth".format(n_rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def get_metrics_for_classification(self, data_loader, model):

        model.eval()

        truths = []
        preds = []
        losses = []
        
        for batch in tqdm(data_loader, mininterval=1):
            data = batch
            data = data.cuda()
            
            # Forward pass through model (no sampler)
            if data.edge_attr is not None:
                out = model(data.x, data.edge_index, data.batch, data.edge_attr)
            else:
                out = model(data.x, data.edge_index, data.batch)
            
            # Get labels and predictions
            truths += data.y.cpu().numpy().tolist()
            preds += out.detach().cpu().numpy().tolist()
            
            # Calculate loss based on loss type
            if self.params.loss == 'CrossEntropyLoss':
                # Multi-class: use one-hot encoded labels
                labels = torch.nn.functional.one_hot(data.y, num_classes=out.shape[1])
                loss = self.criterion(out, labels.float())
            else:
                # Binary: BCEWithLogitsLoss expects raw logits and float labels
                loss = self.criterion(out.squeeze(), data.y.float())
            
            losses.append(loss.item())

        truths = np.array(truths)
        preds = np.array(preds)
        
        # Get predicted classes based on loss type
        if self.params.loss == 'CrossEntropyLoss':
            # Multi-class: argmax over classes
            preds_class = np.argmax(preds, axis=1)
            preds_proba = torch.softmax(torch.tensor(preds), dim=1).numpy()
        else:
            # Binary: sigmoid threshold at 0.5
            preds_proba = torch.sigmoid(torch.tensor(preds)).numpy()
            if len(preds_proba.shape) > 1:
                preds_proba = preds_proba.squeeze()
            preds_class = (preds_proba > 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(truths, preds_class)
        f1 = f1_score(truths, preds_class, average='binary' if self.params.loss != 'CrossEntropyLoss' else 'macro', zero_division=0)
        precision = precision_score(truths, preds_class, average='binary' if self.params.loss != 'CrossEntropyLoss' else 'macro', zero_division=0)
        recall = recall_score(truths, preds_class, average='binary' if self.params.loss != 'CrossEntropyLoss' else 'macro', zero_division=0)
        
        # AUC-ROC
        try:
            if self.params.loss == 'CrossEntropyLoss':
                # Multi-class AUC
                auc = roc_auc_score(truths, preds_proba, multi_class='ovr', average='macro')
            else:
                # Binary AUC
                auc = roc_auc_score(truths, preds_proba)
        except:
            auc = 0.0
        
        loss = np.mean(losses)
        
        return acc, f1, precision, recall, auc, loss    

    def train_for_classification(self):
        acc_best = 0
        f1_best = 0
        loss_best = 10000
        epochs_no_improve = 0

        for epoch in range(self.params.epochs):
            
            self.model.train()
            start_time = timer()
            losses = []
            truths = []
            preds = []
            
            for batch in tqdm(self.data_loader['train'], mininterval=10):
                data = batch
                data = data.cuda()
                self.optimizer.zero_grad()

                # Forward pass through model (no sampler)
                if data.edge_attr is not None:
                    out = self.model(data.x, data.edge_index, data.batch, data.edge_attr)
                else:
                    out = self.model(data.x, data.edge_index, data.batch)
                
                # Get labels
                if self.params.loss == 'CrossEntropyLoss':
                    # Multi-class: use one-hot encoded labels
                    labels = torch.nn.functional.one_hot(data.y, num_classes=out.shape[1])
                    loss = self.criterion(out, labels.float())
                else:
                    # Binary: BCEWithLogitsLoss expects raw logits and float labels
                    loss = self.criterion(out.squeeze(), data.y.float())
                
                truths += data.y.detach().cpu().numpy().tolist()
                preds += out.detach().cpu().numpy().tolist()
                
                loss.backward()
                losses.append(loss.item())
                
                if hasattr(self.params, 'clip_value') and self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                
                self.optimizer.step()

            optim_state = self.optimizer.state_dict()

            truths = np.array(truths)
            preds = np.array(preds)
            
            # Classification metrics based on loss type
            if self.params.loss == 'CrossEntropyLoss':
                # Multi-class: argmax over classes
                t_preds_class = np.argmax(preds, axis=1)
                t_preds_proba = torch.softmax(torch.tensor(preds), dim=1).numpy()
            else:
                # Binary: sigmoid threshold at 0.5
                t_preds_proba = torch.sigmoid(torch.tensor(preds)).numpy()
                if len(t_preds_proba.shape) > 1:
                    t_preds_proba = t_preds_proba.squeeze()
                t_preds_class = (t_preds_proba > 0.5).astype(int)
            
            t_acc = accuracy_score(truths, t_preds_class)
            t_f1 = f1_score(truths, t_preds_class, average='binary' if self.params.loss != 'CrossEntropyLoss' else 'macro', zero_division=0)
            t_precision = precision_score(truths, t_preds_class, average='binary' if self.params.loss != 'CrossEntropyLoss' else 'macro', zero_division=0)
            t_recall = recall_score(truths, t_preds_class, average='binary' if self.params.loss != 'CrossEntropyLoss' else 'macro', zero_division=0)
            
            # For AUC-ROC
            try:
                if self.params.loss == 'CrossEntropyLoss':
                    # Multi-class AUC
                    t_auc = roc_auc_score(truths, t_preds_proba, multi_class='ovr', average='macro')
                else:
                    # Binary AUC
                    t_auc = roc_auc_score(truths, t_preds_proba)
            except:
                t_auc = 0.0

            with torch.no_grad():
                v_acc, v_f1, v_precision, v_recall, v_auc, v_loss = self.get_metrics_for_classification(
                    self.data_loader['val'], self.model
                )

                print(
                    "Epoch {} : Training Loss: {:.5f}, Acc: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, AUC: {:.4f}, LR: {:.6f}, Time: {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        t_acc,
                        t_f1,
                        t_precision,
                        t_recall,
                        t_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(
                    "Val Loss: {:.5f}, Acc: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, AUC: {:.4f}".format(
                        v_loss,
                        v_acc,
                        v_f1,
                        v_precision,
                        v_recall,
                        v_auc
                    )
                )
                
                # Save best model based on F1 score
                if v_f1 > f1_best: 
                    print("Val F1 improving....saving weights !! ")

                    best_epoch = epoch + 1
                    acc_best = v_acc
                    f1_best = v_f1
                    loss_best = v_loss

                    self.best_model_states = copy.deepcopy(self.model.state_dict())

                    epochs_no_improve = 0

                    if not os.path.isdir(self.params.model_dir):
                        os.makedirs(self.params.model_dir)

                    # Save model
                    model_path = self.params.model_dir + "/model_epoch{}_F1_{:.4f}.pth".format(best_epoch, f1_best)
                    torch.save(self.model.state_dict(), model_path)
                    print(f"Model saved in {model_path}")

                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= self.params.patience:
                    print(f"Early stopping at epoch {epoch + 1}. Best Val F1: {f1_best:.4f}, Best Val Acc: {acc_best:.4f} (epoch {best_epoch})")
                    break

        # Load best model
        self.model.load_state_dict(self.best_model_states)
        print(f"Restored best model from epoch {best_epoch}")

        # Test evaluation
        with torch.no_grad():
            print("***************************Test************************")
            test_acc, test_f1, test_precision, test_recall, test_auc, test_loss = self.get_metrics_for_classification(
                self.data_loader['test'], self.model
            )
            print("***************************Test Results************************")
            print(
                "Test Evaluation: Loss: {:.5f}, Acc: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, AUC: {:.4f}".format(
                    test_loss,
                    test_acc,
                    test_f1,
                    test_precision,
                    test_recall,
                    test_auc
                )
            )

            # Save final test results
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            
            final_model_path = self.params.model_dir + "/final_model_testF1_{:.4f}.pth".format(test_f1)
            torch.save(self.model.state_dict(), final_model_path)
            
            print(f"Final model saved in {final_model_path}")
        
        # Return comprehensive results
        results = {
            'best_epoch': best_epoch,
            'best_val_acc': acc_best,
            'best_val_f1': f1_best,
            'best_val_loss': loss_best,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'test_loss': test_loss
        }
        
        return results
