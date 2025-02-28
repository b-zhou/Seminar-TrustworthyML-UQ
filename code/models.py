import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

class MLP:
    def __init__(self, input_dim=2, hidden1_dim=8, hidden2_dim=8, output_type="binary", dropout_rate=0.5):
        super(MLP, self).__init__()

        self.output_type = output_type
        criteria = {
            "binary": nn.BCEWithLogitsLoss(),
            "numeric": nn.MSELoss(),
            "mean+sd": nn.GaussianNLLLoss(),
        }
        self.criterion = criteria[self.output_type]

        if dropout_rate == 0:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden1_dim),
                nn.ReLU(),
                nn.Linear(hidden1_dim, hidden2_dim),
                nn.ReLU(),
                nn.Linear(hidden2_dim, 1 if output_type in ["binary", "numeric"] else 2)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden1_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden1_dim, hidden2_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden2_dim, 1 if output_type in ["binary", "numeric"] else 2)
            )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model.to(self.device)

        self.training_loss = []
        self.accuracy = []
    
    # def init_params(self, seed=0):
    #     torch.manual_seed(seed)
    #     # define helper function for weight initialization
    #     def init_layer(layer):
    #         if isinstance(layer, nn.Linear):
    #             nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    #             nn.init.zeros_(layer.bias)
    #     # apply to all layers
    #     self.model.apply(init_layer)

    def loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)
    
    def loss_gaussian(self, out, y_true):
        mean = out[:, 0]
        var = torch.exp(out[:, 1])
        return self.criterion(mean, y_true, var)
    
    def train(
            self, train_loader,
            n_epochs=1000, lr=0.5, weight_decay=1e-5,
            milestones=[500], gamma=0.1,
            seed=0, log_interval=10
        ):
        model = self.model
        model.to(self.device)
        torch.manual_seed(seed)

        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        loss_fn = self.loss_gaussian if self.output_type == "mean+sd" else self.loss

        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            acc = 0 if self.output_type == "binary" else torch.nan

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # forward
                y_pred = model(x_batch)
                batch_loss = loss_fn(y_pred, y_batch)

                # backward
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # by default, the loss function computes the average loss over the batch
                # so we multiply by the batch size to get the total loss
                train_loss += batch_loss.item() * x_batch.size(0)
                # for classification, also compute accuracy
                if self.output_type in "binary":
                    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
                    acc += (y_pred == y_batch).sum().item()
            
            # average loss over the entire training set
            train_loss /= len(train_loader.dataset)
            acc /= len(train_loader.dataset)
            self.training_loss.append(train_loss)
            self.accuracy.append(acc)
            # print loss
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.6f}, Accuracy: {acc:.6f}")
            
            scheduler.step()

    def predict(self, x, return_type = "numpy", mc_dropout = False):
        """
        Predict the output for an input array (batch) x
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)
        model = self.model
        if mc_dropout:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            y_pred = model(x)
            if self.output_type == "binary":
                y_pred = torch.sigmoid(y_pred)
                if return_type == "numpy":
                    return y_pred.cpu().numpy()
                return y_pred
            if self.output_type == "mean+sd":
                mean = y_pred[:, 0]
                sd = torch.exp(y_pred[:, 1])
                if return_type == "numpy":
                    return mean.cpu().numpy(), sd.cpu().numpy()
                return mean, sd
            # numeric
            if return_type == "numpy":
                return y_pred.cpu().numpy()
            return y_pred