# Install libraries
import os
import sys
import time

import torch
from torch import nn
import torch_geometric

from sklearn.model_selection import ParameterGrid
import torch.optim as optim

import torch_sparse

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

# Folders

main_folder = '/path/main_folder/'
dataset_folder = '/path/Data/'
results_folder =  'path/save_results'

# Dataset
sys.path.insert(1, os.path.join(main_folder, 'utils'))

from customDataset import BrainGraphDataset
from kFoldDataLoader import KFold_DataLoader
from EarlyStopping2 import EarlyStopping2


# Models 

sys.path.insert(0, os.path.join(main_folder, 'models'))
from GCN import GCN
from DEP import DEP


# Config dictionary files for each model and for training
# In here ParameterGrid is used only when performing hyperparameter tunning -> serves to select a parameter combination from the grid

config_gcn = ParameterGrid({'num_layers': [4], 
                            'hidden_dim': [256], 
                            'edge_attr': [1], # 1, None: for unweighted graphs = None
                            'activ_funct': [nn.ReLU()],
                            'dropout': [0.5], 
                            'final_dropout': [None], 
                            'graph_pooling_type': ['global_mean_pool']}
                            )[0]


config_training = ParameterGrid({'epochs': [1000], #10000
                                 'batch': [16],
                                 'lr': [0.001],
                                 'weight_decay': [None]
                                 })[0]


# Cuda
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:3")  # Use GPU 0
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")


# random seed setting
seed =  42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Dataset 
dataset_name = 'NodeID_EdgeW_PearC_Sp5.5_raw_Sex'#'PearC_EdgeW_PearC_Sp30_raw_Sex' # Choose type of dataset: Sex, Handedness
hdf5_path =os.path.join(dataset_folder, f"{dataset_name}.h5") 
dataset_dir = os.path.join(dataset_folder, f"{dataset_name}_Dataset")
dataset = BrainGraphDataset(root=Path(dataset_dir), hdf5_path=hdf5_path )

#  Split Data  Train | Test
train_size = int(len(dataset)*0.9)

dataset = dataset.shuffle()
#train_dataset = dataset[:train_size]
#test_dataset = dataset[train_size:]

#train_loader = DataLoader(train_dataset, batch_size=config_training['batch'], shuffle=True)
#val_loader = DataLoader(test_dataset, batch_size=config_training['batch'], shuffle=False)

sfk = KFold_DataLoader(num_repeats=1, num_k=4, batch_size=config_training['batch'], stratify= True, random_seed=42)
train_loaders, val_loaders = sfk.get_nk_loaders(dataset)

# Data dimensions
edge_feat_dim = dataset[0].edge_attr.shape[0]
output_dim = len(dataset[0].y.shape) +2 
node_feat_dim = dataset.num_node_features


# The models

model = GCN(config_gcn, input_dim=node_feat_dim, output_dim=output_dim)

    
criterion = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()


def training(model, criterion):

    t = time.time()
    model.train()
    sum_loss = []
    

    for data in train_loader:

        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch, data.edge_attr)


        labels = torch.nn.functional.one_hot(data.y, num_classes=2) # for cross_entropy
        loss = criterion(out, labels.float())

        sum_loss.append(loss.item())
        loss.backward()
 
        optimizer.step()



    avg_loss = np.mean(sum_loss)

    train_t = time.time() - t

    return ( avg_loss.item(), train_t )



def evaluate(loader, model, criterion ):
    
    t = time.time()
    model.eval()
    losses = []
    y_true = []
    y_pred = []

    for data in loader:

        data = data.to(device)

         
        out = model(data.x, data.edge_index, data.batch, data.edge_attr)


        labels = torch.nn.functional.one_hot(data.y, num_classes=2) # for cross_entropy
        loss = criterion(out, labels.float())

        #pred = torch.sigmoid(out)
        pred = out.argmax(dim=1)

        losses.append(loss.item()) #+ l2_penalty.item()
        y_pred.append(pred.detach().cpu().numpy())
        y_true.append(data.y.detach().cpu().numpy())

    eval_t = time.time() - t

    avg_loss = np.mean(losses)
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)

    #y_bin = (y_pred >0.5).astype(np.float32)

    clr = classification_report( y_true, y_pred, output_dict=True)


    return (avg_loss, clr, eval_t)

######################################################################################### MODEL TRAINING #####################################################################################################

fold = 0

for train_loader, val_loader in zip(train_loaders, val_loaders):
    fold += 1

    print( "Fold {} out of 4".format(fold))

    # The sampler
    model = GCN(config_gcn, input_dim=node_feat_dim, output_dim=output_dim)

    optimizer = optim.Adam(model.parameters(), lr=config_training['lr'] )
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma= 0.8) #initial_lr=0.01,gamma= 0.97

    model = model.to(device)
    early_stopping0 = EarlyStopping2(patience=50, delta=0.001)


    for epoch in range(0, config_training['epochs']):
        

        outputs = training( model,  criterion)

        #scheduler.step()

        train_outputs = evaluate(train_loader, model, criterion)
        val_outputs = evaluate(val_loader, model, criterion )

        f1score_train=train_outputs[1]['macro avg']['f1-score']
        f1score_val=val_outputs[1]['macro avg']['f1-score']
        acc_train=train_outputs[1]['accuracy']
        acc_val=val_outputs[1]['accuracy']

        print('Epoch: {:04d}'.format(epoch + 1),
            'loss_train: {:.4f}'.format(train_outputs[0]),
            'loss_val: {:.4f}'.format(val_outputs[0]),
            'f1score_train: {:.4f}'.format(f1score_train),
            'f1score_val: {:.5f}'.format(f1score_val))
        
        save_model = {'train_loss': train_outputs[0],'val_loss': val_outputs[0],
                        'train_f1': f1score_train, 'val_f1': f1score_val, 'train_acc': acc_train, 
                        'val_acc': acc_val
                    }

        if early_stopping0(model, val_outputs[0], save_model, epoch+1):
                
            print(f"Early stopping at epoch {epoch+1}")

            early_stopping0.restore_best_model(model)

            break


    # Dictionary to store results

    # if the early stopping0 criteria was activated : save that sampler result
    if early_stopping0.early_stop == True:
        results_dict = early_stopping0._store_criteria
    else: # otherwise save last sampler results
        results_dict = save_model
    
    # Convert dict to DataFrame (single-row)
    df = pd.DataFrame([results_dict])

    # Check if file exists to determine whether to write header
    results_filename = '/path/training_results.csv'
    write_header = not os.path.exists(results_filename)

    # Append row to CSV
    df.to_csv(results_filename, mode='a', header=write_header, index=False)

    torch.save(model.state_dict(), f"/path/model_{fold}.pth")
