import torch
import copy
import torch.nn.utils.prune as prune


class EarlyStopping2:
    def __init__(self,  patience=5, delta=0.0):
        self.patience = patience  # Number of epochs to wait for improvement
        self.delta = delta  # Minimum change to qualify as an improvement
        #self.restore_best_model = restore_best_model 
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_model, val_loss, store_criteria, epoch):
        if self.best_score is None:
            self.best_score = val_loss
            self._best_model_weights = self._get_model_copy(current_model)
            self._store_criteria = store_criteria
            self._epoch = epoch
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0  # Reset counter when improvement happens
            self._best_model_weights = self._get_model_copy(current_model)
            self._store_criteria = store_criteria
            self._epoch = epoch
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    
    def _get_model_copy(self, model):
        """Deep copy the model by copying its state_dict."""

        if isinstance(model, list):

            best_model_weights = []

            for modeli in model: # if we need to save more than 1 best model

                if hasattr(modeli, "weight_mask_orig"): 
                    #prune.remove(model.lin, "weight")  # Remove old pruning mask  
                    modeli.weight_mask_orig.data.copy_( modeli.weight_mask) 
                    best_model_weights.append( copy.deepcopy(modeli.state_dict()))

                else:
                    best_model_weights.append(copy.deepcopy(modeli.state_dict()) )

        else:
            best_model_weights = copy.deepcopy(model.state_dict()) 

        return best_model_weights
    
    def reset_state(self):

        self.counter = 0
        self.early_stop = False
        self.best_score = None
        

    def restore_best_model(self, model):
        """Returns the best model stored during training."""
        # Load the best model weights

        if isinstance(self._best_model_weights, list):
                
            for modeli, best_model_weights in zip(model, self._best_model_weights):

                if hasattr(modeli, "weight_mask_orig"):
                    modeli.load_state_dict(best_model_weights)
                    prune.remove(modeli, "weight_mask")  

                else:
                    modeli.load_state_dict(best_model_weights) # modified in place

        
        else:
                model.load_state_dict(self._best_model_weights) # modified in place

