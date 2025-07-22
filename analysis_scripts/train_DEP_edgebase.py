# Install libraries
import os
import sys
import time

import torch
from torch import nn
import torch_geometric
import torch_sparse
import torch_geometric.datasets as Datasets
import torch_geometric.data as Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as transforms
import torch.optim as optim

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import ParameterGrid
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
from GINE import GINE
from GAT import xGAT
from DEP import DEP


# Config dictionary files for each model and for training
# In here ParameterGrid is used only when performing hyperparameter tunning -> serves to select a parameter combination from the grid

config_gat = ParameterGrid({'num_layers': [2], 
                            'hidden_dim': [32], 
                            'final_hidden_dim': [32],
                            'num_heads' : [4],
                            'num_heads_final' : [4],
                            #'input_dim': [dataset[0].num_node_features],
                            #'output_dim':  [dataset.num_classes],
                            'edge_attr': [1], #[dataset[0].edge_attr.view(-1,1).shape[1]], # for unweighted graphs = None
                            'activ_funct': [nn.ReLU()], 
                            'final_dropout': [0], #0.5
                            'graph_pooling_type': ['global_mean_pool']}
                            )[0]

config_gin = ParameterGrid({'num_layers': [2], 
                            'num_mlp_layers': [1],
                            'hidden_dim': [32], 
                            'edge_attr': [1], # 1, None [dataset[0].edge_attr.view(-1,1).shape[1]], # for unweighted graphs = None
                            'activ_funct': [nn.ReLU()], 
                            'final_dropout': [0.5], 
                            'graph_pooling_type': ['global_mean_pool'], #'global_mean_pool'
                            'train_eps': [True],
                            })[0]

config_gcn = ParameterGrid({'num_layers': [4], 
                            'hidden_dim': [256], 
                            #'input_dim': [dataset[0].num_node_features],
                            #'output_dim':  [dataset.num_classes],
                            'edge_attr': [1], # 1, None [dataset[0].edge_attr.view(-1,1).shape[1]], # for unweighted graphs = None
                            'activ_funct': [nn.ReLU()], 
                            'final_dropout': [0.5], 
                            'graph_pooling_type': ['global_mean_pool']}
                            )[0]
                        
                        

config_sampler = ParameterGrid({'alpha': [0.00001], #0.001
                                'beta': [0.001],
                                'min_sp': [0.05],
                                'lr': [None], # 0.0001
                                'dropout': [None], #[None] #[0.5] 
                                'ite_step': [9], #4
                                'init_prune_step': [0.05]
                                })[0]

config_training = ParameterGrid({'epochs': [1000], #10000
                                 'batch': [16],
                                 'lr': [0.001],
                                 'weight_decay': [None]
                                 })[0]


# Cuda
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use GPU 0
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")


# random seed setting
seed =  42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# PRE TRANSFORMS?

# OPTION 1: Using pre_transform during dataset creation
def adjust_labels(data):
    """Pre-transform function to adjust labels"""
    # Handle different label scenarios
    if data.y.item() == 1:
        data.y = torch.tensor(0)
    elif data.y.item() == 2:
        data.y = torch.tensor(1)
    # Add more cases if needed:
    # elif data.y.item() == 3:
    #     data.y = torch.tensor(2)
    
    return data

# Dataset 
dataset_name = 'NodeID_EdgeW_PearC_Sp_fully_connected_raw_Handedness_class' #'NodeID_EdgeW_PearC_Sp_fully_connected_raw_ReadEng_AgeAdj_class', 'NodeID_EdgeW_PearC_Sp_fully_connected_raw_Sex'#'PearC_EdgeW_PearC_Sp_fully_connected_raw_Sex' # Choose type of dataset: Sex, Handedness
hdf5_path =os.path.join(dataset_folder, f"{dataset_name}.h5") 
dataset_dir = os.path.join(dataset_folder, f"{dataset_name}_Dataset")
#dataset = BrainGraphDataset(root=Path(dataset_dir), hdf5_path=hdf5_path )

#WITH TRANSFORM :
dataset = BrainGraphDataset(root=Path(dataset_dir), hdf5_path=hdf5_path,pre_transform=adjust_labels )

#  Split Data  Train | Test
train_size = int(len(dataset)*0.8)

dataset = dataset.shuffle()
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]


sfk = KFold_DataLoader(num_repeats=1, num_k=4, batch_size=config_training['batch'], stratify= True, random_seed=42)
train_loaders, val_loaders = sfk.get_nk_loaders(dataset)

test_loader = DataLoader(test_dataset, batch_size=config_training['batch'], shuffle=False)


# Data dimensions
edge_feat_dim = dataset[0].edge_attr.shape[0]
output_dim = len(dataset[0].y.shape) +2 
node_feat_dim = dataset.num_node_features

# The sampler
sampler = DEP( node_feat_dim, output_dim, config_sampler) # the input_dim is always the same as the number of ROIs or number of row/column of original adjancecy


# The models
#model = xGINE(config_gin, input_dim=node_feat_dim, output_dim=output_dim)
#GATmodel = xGAT(config_gat, input_dim=node_feat_dim, output_dim=output_dim)
model = GCN(config_gcn, input_dim=node_feat_dim, output_dim=output_dim)

# Optimizer
if config_training['weight_decay'] is not None:
    pipeline = list(sampler.parameters()) + list(model.parameters())
    optimizer = optim.Adam(pipeline, lr=config_training['lr'], weight_decay=config_training['weight_decay'] )
else:
    pipeline = list(sampler.parameters()) + list(model.parameters())
    optimizer = optim.Adam(pipeline, lr=config_training['lr'])


scheduler = None


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
criterion = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()



def training(sampler, model, criterion):

    t = time.time()
    sampler.train()
    model.train()
    sum_loss = []
    

    for data in train_loader:

        data = data.to(device)
        optimizer.zero_grad()


        sampled_data, _ = sampler(data)

        out = model(sampled_data.x, sampled_data.edge_index, sampled_data.batch, sampled_data.edge_attr)


        labels = torch.nn.functional.one_hot(data.y, num_classes=2) # for cross_entropy
        loss_clf = criterion(out, labels.float())
        l1_penalty = sampler.l1_norm()
        #l2_penalty = sampler.l2_norm()
        loss = loss_clf + l1_penalty #+ l2_penalty # for elastic net

        sum_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        

    avg_loss = np.mean(sum_loss)

    train_t = time.time() - t

    return ( avg_loss.item(), train_t )



def evaluate(loader, sampler, model, criterion ):
    
    t = time.time()
    sampler.eval()
    model.eval()
    losses = []
    y_true = []
    y_pred = []

    for data in loader:

        data = data.to(device)

        sampled_data, mask = sampler(data) 

         
        out = model(sampled_data.x, sampled_data.edge_index, sampled_data.batch, sampled_data.edge_attr)


        labels = torch.nn.functional.one_hot(data.y, num_classes=2) # for cross_entropy
        loss = criterion(out, labels.float())
        l1_penalty = sampler.l1_norm()
        #l2_penalty = sampler.l2_norm()

        #pred = torch.sigmoid(out)
        pred = out.argmax(dim=1)

        losses.append(loss.item() + l1_penalty.item()) #+ l2_penalty.item()
        y_pred.append(pred.detach().cpu().numpy())
        y_true.append(data.y.detach().cpu().numpy())

    eval_t = time.time() - t

    avg_loss = np.mean(losses)
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)

    #y_bin = (y_pred >0.5).astype(np.float32)

    clr = classification_report( y_true, y_pred, output_dict=True)


    return (mask, avg_loss, clr, eval_t)

######################################################################################### MODEL TRAINING #####################################################################################################

# Early Stopping

fold = 0

for train_loader, val_loader in zip(train_loaders, val_loaders):


    fold += 1

    if fold!=4:
        continue

    print( "Fold {} out of 4".format(fold))

    # The sampler
    sampler = DEP(node_feat_dim, output_dim, config_sampler)
    model = GCN(config_gcn, input_dim=node_feat_dim, output_dim=output_dim)

    # Optimizer
    if config_training['weight_decay'] is not None:
        pipeline = list(sampler.parameters()) + list(model.parameters())
        optimizer = optim.Adam(pipeline, lr=config_training['lr'], weight_decay=config_training['weight_decay'] )
    else:
        pipeline = list(sampler.parameters()) + list(model.parameters())
        optimizer = optim.Adam(pipeline, lr=config_training['lr'])

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma= 0.8) #initial_lr=0.01,gamma= 0.97

    sampler = sampler.to(device)
    model = model.to(device)
    early_stopping0 = EarlyStopping2(patience=50, delta=0.001)
    step = config_sampler['init_prune_step']


    for epoch in range(0, config_training['epochs']):
        

        outputs = training(sampler, model,  criterion)

        #scheduler.step()

        train_outputs = evaluate(train_loader, sampler, model, criterion)
        val_outputs = evaluate(val_loader,sampler, model, criterion )

        edge_reduction = round((np.where(val_outputs[0]==False)[0].shape[0] /val_outputs[0].shape[0])*100,2)
        f1score_train=train_outputs[2]['macro avg']['f1-score']
        f1score_val=val_outputs[2]['macro avg']['f1-score']
        acc_train=train_outputs[2]['accuracy']
        acc_val=val_outputs[2]['accuracy']

        #weights = sampler.get_params()
        print('Epoch: {:04d}'.format(epoch + 1),
            'Edge Reduction: {:.2f} %'.format(edge_reduction),
            'loss_train: {:.4f}'.format(train_outputs[1]),
            'loss_val: {:.4f}'.format(val_outputs[1]),
            'f1score_train: {:.4f}'.format(f1score_train),
            'f1score_val: {:.5f}'.format(f1score_val))
        
        save_sampler = {'edge_reduction': edge_reduction, 'train_loss': train_outputs[1],'val_loss': val_outputs[1],
                        'train_f1': f1score_train, 'val_f1': f1score_val, 'train_acc': acc_train, 
                        'val_acc': acc_val
                        }

        if early_stopping0([sampler, model], val_outputs[1], save_sampler, epoch+1):
                
            print(f"Early stopping at epoch {epoch+1}")

            early_stopping0.restore_best_model([sampler, model])
            edge_reduction = early_stopping0._store_criteria['edge_reduction'] 
            best_epoch = early_stopping0._epoch
            break

        if (epoch % config_sampler['ite_step'] ==0) & (epoch!=0): 

        
            if (edge_reduction >= 90) & (edge_reduction < 99):
                step = 0.01

            elif (edge_reduction >= 99) & (edge_reduction != 100):
                step = 0.001

            elif edge_reduction == 100:
                early_stopping0.early_stop = True
                early_stopping0.restore_best_model([sampler, model])
                edge_reduction = early_stopping0._store_criteria['edge_reduction']
                best_epoch = early_stopping0._epoch
                break        

            with torch.no_grad():
                sampler.update_prune(step)
        

################## TEST TRAINED MODEL #####################################################################################################################
    model.eval()
    y_true = []
    y_preds = []
    num_batches = 0

    #  In first model
    for data in test_loader:
            
            data = data.to(device)

            sampled_data, mask =  sampler(data) 

            out, alpha = model(sampled_data.x, sampled_data.edge_index, sampled_data.batch, sampled_data.edge_attr)
            pred = out.argmax(dim=1)

            num_batches += 1  # Count batches
            y_true.append(list(data.y.cpu().numpy()))
            y_preds.append(list(pred.detach().cpu().numpy()))


    y_true = np.hstack(y_true)
    y_preds = np.hstack(y_preds)

    #out = root_mean_squared_error(y_true, y_preds)
    #y_bin = (y_preds >0.5).astype(np.float32)

    out = classification_report(y_true, y_preds, output_dict=True, zero_division=0.0)

    test_results1 = {'orig_model_test_f1': out['macro avg']['f1-score'],'orig_model_test_acc': out['accuracy']}

    print(f"Orig Model: Test f1: {test_results1['orig_model_test_f1']:.4f}, Test Acc: {test_results1['orig_model_test_acc']:.4f}")
    

    # Dictionary to store results

    # if the early stopping0 criteria was activated : save that sampler result
    if early_stopping0.early_stop == True:
        sampler_results = early_stopping0._store_criteria
    else: # otherwise save last sampler results
        sampler_results = save_sampler 

    results = sampler_results | test_results1


    # Convert dict to DataFrame (single-row)
    df = pd.DataFrame([results])

    # Check if file exists to determine whether to write header
    results_filename = '/path/save_results/DEP_GAT_edges_fully_cognfluidunadj_best_model_training_results.csv'
    write_header = not os.path.exists(results_filename)
    # Append row to CSV
    df.to_csv(results_filename, mode='a', header=write_header, index=False)


    weights_table = pd.DataFrame(mask.int()).T
    # Check if file exists to determine whether to write header
    wtable_filename = '/path/save_results/DEP_GAT_edges_fully_cognfluidunadj_best_model_weights_table.csv'
    write_header = not os.path.exists(wtable_filename)
    weights_table.to_csv(wtable_filename, mode='a', header=write_header, index=False)

    # Check if file exists to determine whether to write header
    #alpha_table = pd.DataFrame(np.hstack(alpha.detach().cpu().numpy())).T
    alpha_table = pd.DataFrame(alpha.detach().cpu().numpy()).T
    #alpha.detach().cpu().numpy().shape[0] shape x 4

    atable_filename = '/path/save_results/DEP_GAT_edges_fully_cognfluidunadj_best_model_alpha_table.csv'
    write_header = not os.path.exists(atable_filename)
    alpha_table.to_csv(atable_filename, mode='a', header=write_header, index=False)

    #torch.save(sampler.state_dict(), f"/path/save_results/sampler_DEP_{fold}.pth")
    #torch.save(model.state_dict(), f"/path/save_results/classifier_DEP_{fold}.pth")





"""
######################################################################################### MODEL TRAINING #####################################################################################################

# Early Stopping

fold = 0

for train_loader, val_loader in zip(train_loaders, val_loaders):


    fold += 1

    if fold!=4:
        continue

    print( "Fold {} out of 4".format(fold))

    # The sampler
    sampler = IGS(379, output_dim, config_sampler)
    model = xGAT(config_gat, input_dim=node_feat_dim, output_dim=output_dim)

    pipeline = list(sampler.parameters()) + list(model.parameters())
    optimizer = optim.Adam(pipeline, lr=config_training['lr'] )
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma= 0.8) #initial_lr=0.01,gamma= 0.97


    sampler = sampler.to(device)
    model = model.to(device)
    early_stopping0 = EarlyStopping2(patience=50, delta=0.001)
    step = config_sampler['init_prune_step']


    for epoch in range(0, config_training['epochs']):
        

        outputs = training(sampler, model,  criterion)

        #scheduler.step()

        train_outputs = evaluate(train_loader, sampler, model, criterion)
        val_outputs = evaluate(val_loader,sampler, model, criterion )

        edge_reduction = round((np.where(val_outputs[0]==False)[0].shape[0] /val_outputs[0].shape[0])*100,2) #MASK
        
        f1score_train=train_outputs[2]['macro avg']['f1-score']
        f1score_val=val_outputs[2]['macro avg']['f1-score']
        acc_train=train_outputs[2]['accuracy']
        acc_val=val_outputs[2]['accuracy']

        #weights = sampler.get_params()

        print('Epoch: {:04d}'.format(epoch + 1),
            'Edge Reduction: {:.2f} %'.format(edge_reduction),
            'loss_train: {:.4f}'.format(train_outputs[1]),
            'loss_val: {:.4f}'.format(val_outputs[1]),
            'f1score_train: {:.4f}'.format(f1score_train),
            'f1score_val: {:.5f}'.format(f1score_val))
        
        save_sampler = {'edge_reduction': edge_reduction, 'train_loss': train_outputs[1],'val_loss': val_outputs[1],
                        'train_f1': f1score_train, 'val_f1': f1score_val, 'train_acc': acc_train, 
                        'val_acc': acc_val
                        }

        if early_stopping0([sampler, model], val_outputs[1], save_sampler, epoch+1):
                
            print(f"Early stopping at epoch {epoch+1}")

            early_stopping0.restore_best_model([sampler, model])
            edge_reduction = early_stopping0._store_criteria['edge_reduction'] 
            best_epoch = early_stopping0._epoch
            alpha = val_outputs[4]
            break

        if (epoch % config_sampler['ite_step'] ==0) & (epoch!=0): 

        
            if (edge_reduction >= 90) & (edge_reduction < 99):
                step = 0.01

            elif (edge_reduction >= 99) & (edge_reduction != 100):
                step = 0.001

            elif edge_reduction == 100:
                early_stopping0.early_stop = True
                early_stopping0.restore_best_model([sampler, model])
                edge_reduction = early_stopping0._store_criteria['edge_reduction']
                best_epoch = early_stopping0._epoch
                alpha = val_outputs[4]

                break        

            with torch.no_grad():

                #update_sp = config_sampler['min_sp'] + 0.5
                #prune.remove(sampler, "weight_mask")  # Remove old pruning mask  
                #prune.l1_unstructured(sampler, name="weight_mask", amount=update_sp)  # Reapply pruning
                sampler.update_prune(step)

    ################################################################################ RE-TRAIN MODEL #####################################################################################################################


    model1 = xGAT(config_gat, input_dim=node_feat_dim, output_dim=output_dim)

    model1 = model1.to(device)

    sampler.min_sp = edge_reduction / 100  # reset best edge reduction

    optimizer_model = optim.Adam(model1.parameters(), lr=config_training['lr'])
    #scheduler = optim.lr_scheduler.StepLR(optimizer_model, step_size= 5, gamma= 0.8) #initial_lr=0.01,gamma= 0.97


    # RETRAIN MODEL with sample data

    early_stopping1 = EarlyStopping2(patience=50, delta=0.001)

    for epoch in range(0, config_training['epochs']):

        model1.train()

        for data in train_loader:
                
                # SAMPLING
                data = data.to(device)

                sampled_data, mask =  sampler(data) 

                out = model1(sampled_data.x, sampled_data.edge_index, sampled_data.batch,sampled_data.edge_attr)
                
                loss_clf = criterion(out, data.y) # CROSS ENTROPY LOSS USES *output as LOGITS* and * y_true as a integer of class indices starting from 0 e.g[0,1, 2]
                loss_clf.backward()
                optimizer_model.step()  
                optimizer_model.zero_grad()  
        
        #scheduler.step()

        # MODEL EVALUATION FOR TRAINING SET
        model1.eval()
        y_true = []
        y_preds = []
        losses = []
        num_batches = 0
        total_loss_clf  = 0

        for data in train_loader:
            
            data = data.to(device)

            sampled_data, mask =  sampler(data) 

            out = model1(sampled_data.x, sampled_data.edge_index, sampled_data.batch, sampled_data.edge_attr)
            
            loss_clf = criterion(out, data.y) # CROSS ENTROPY LOSS USES *output as LOGITS* and * y_true as a integer of class indices starting from 0 e.g[0,1, 2]
            total_loss_clf += loss_clf.detach().item()  # Accumulate loss

            
            num_batches += 1  # Count 
            
            pred = out.argmax(dim=1)
            y_true.append(list(data.y.cpu().numpy()))
            y_preds.append(list(pred.detach().cpu().numpy()))

        avg_loss_clf = total_loss_clf / num_batches  # Compute average loss
        y_true = np.hstack(y_true)
        y_preds = np.hstack(y_preds)

        #out = root_mean_squared_error(y_true, y_preds)
        #y_bin = (y_preds >0.5).astype(np.float32)

        out = classification_report(y_true, y_preds, output_dict=True, zero_division=0.0)

        train_results = {'2_train_loss': avg_loss_clf ,'2_train_f1': out['macro avg']['f1-score'],'2_train_acc': out['accuracy']}
        
        # MODEL EVALUATION FOR VALIDATION SET

        model1.eval()
        y_true = []
        y_preds = []
        losses = []
        weights = []
        num_batches = 0
        total_loss_clf  = 0
        total_loss = 0

        for data in val_loader:

            data = data.to(device)

            sampled_data, mask =  sampler(data) 

            out = model1(sampled_data.x, sampled_data.edge_index, sampled_data.batch, sampled_data.edge_attr)
            pred = out.argmax(dim=1)

            loss_clf = criterion(out, data.y) # CROSS ENTROPY LOSS USES *output as LOGITS* and * y_true as a integer of class indices starting from 0 e.g[0,1, 2]
            total_loss_clf += loss_clf.detach().item()  # Accumulate loss

            num_batches += 1  # Count batches
            y_true.append(list(data.y.cpu().numpy()))
            y_preds.append(list(pred.detach().cpu().numpy()))

        avg_loss_clf = total_loss_clf / num_batches  # Compute average loss
        y_true = np.hstack(y_true)
        y_preds = np.hstack(y_preds)

        #out = root_mean_squared_error(y_true, y_preds)
        #y_bin = (y_preds >0.5).astype(np.float32)

        out = classification_report(y_true, y_preds, output_dict=True, zero_division=0.0)

        val_results = {'2_val_loss': avg_loss_clf ,'2_val_f1': out['macro avg']['f1-score'],'2_val_acc': out['accuracy']}


        print(f"Epoch: {epoch+1:03d}, Train clfLoss: {train_results['2_train_loss']:.4f}, Val clfLoss:{val_results['2_val_loss']:.4f}, Train f1: {train_results['2_train_f1']:.4f}, Val f1: {val_results['2_val_f1']:.4f}")

        clf_save = train_results | val_results

        if early_stopping1(model1, val_results['2_val_loss'], clf_save, epoch+1):
            print(f"Early stopping at epoch {epoch+1}")
            early_stopping1.restore_best_model(model1)
            break


    # Dictionary to store results

    # if the early stopping0 criteria was activated : save that sampler result
    if early_stopping0.early_stop == True:
        sampler_results = early_stopping0._store_criteria
    else: # otherwise save last sampler results
        sampler_results = save_sampler 

    # if the early stopping1 criteria was activated : save that model results
    if early_stopping1.early_stop == True:
        clf_results = early_stopping1._store_criteria
    else: # otherwise save last model results
        clf_results = clf_save 


    results_dict = sampler_results | clf_results
    
    # Convert dict to DataFrame (single-row)
    df = pd.DataFrame([results_dict])

    # Check if file exists to determine whether to write header
    results_filename = '/path/save_results/DEP_edge/DEP_GAT_edges_fully_sex_best_model_training_results.csv'
    write_header = not os.path.exists(results_filename)

    # Append row to CSV
    df.to_csv(results_filename, mode='a', header=write_header, index=False)

    weights_table = pd.DataFrame(mask.int()).T

    # Check if file exists to determine whether to write header
    wtable_filename = '/path/save_results/DEP_edge/DEP_GAT_edges_fully_sex_best_model_weights_table.csv'
    write_header = not os.path.exists(wtable_filename)
    weights_table.to_csv(wtable_filename, mode='a', header=write_header, index=False)

    # Check if file exists to determine whether to write header
    #alpha_table = pd.DataFrame(np.hstack(alpha.detach().cpu().numpy())).T
    alpha_table = pd.DataFrame(alpha.detach().cpu().numpy()).T
    #alpha.detach().cpu().numpy().shape[0] shape x 4

    atable_filename = '/path/save_results/DEP_GAT_edges_fully_sex_best_model_alpha_table.csv'
    write_header = not os.path.exists(atable_filename)
    alpha_table.to_csv(atable_filename, mode='a', header=write_header, index=False)

    #torch.save(sampler.state_dict(), f"//path/save_results/sampler_IGS_{fold}.pth")
    #torch.save(model.state_dict(), f"/path/save_results/classifier_IGS_{fold}.pth")
"""


