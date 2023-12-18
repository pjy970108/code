# ref:  https://github.com/quantylab/rltrader/blob/master/src/quantylab/rltrader/networks/networks_pytorch.py
import threading
import abc
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001,
                 shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss

        inp = (self.num_steps, input_dim) if hasattr(self, 'num_steps') else (self.input_dim,)
        self.head = self.shared_network if self.shared_network else self.get_network_head(inp, self.output_dim)

        self.model = torch.nn.Sequential(self.head)
        self.add_activation_layer()
        self.model.apply(Network.init_weights)
        self.model.to(device)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        self.criterion = self.get_criterion()

    def add_activation_layer(self):
        if self.activation == 'linear':
            pass
        else:
            activation_layer = getattr(torch.nn, self.activation.capitalize())()
            self.model.add_module('activation', activation_layer)

    def get_criterion(self):
        if self.loss == 'mse':
            return torch.nn.MSELoss()
        elif self.loss == 'binary_crossentropy':
            return torch.nn.BCELoss()

    def predict(self, sample):
        with self.lock:
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(np.array(sample)).float().to(device)
                pred = self.model(x).detach().cpu().numpy().flatten()
            return pred

    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            self.model.train()
            x, y = torch.from_numpy(np.array(x)).float().to(device), torch.from_numpy(np.array(y)).float().to(device)
            y_pred = self.model(x)
            _loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
            loss += _loss.item()
        return loss

    def save_model(self, model_path):
        if model_path and self.model:
            torch.save(self.model, model_path)

    def load_model(self, model_path):
        if model_path:
            self.model = torch.load(model_path)

    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
        if net == 'dnn':
            return DNN.get_network_head((input_dim,), output_dim)

    @abc.abstractmethod
    def get_network_head(self, inp, output_dim):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
            torch.nn.init.normal_(m.weight, std=0.01)


class DNN(Network):
    def get_network_head(self, inp, output_dim):
        return torch.nn.Sequential(
            *self.create_block(inp[0], 256),
            torch.nn.Linear(256, output_dim)
        )

    def create_block(self, input_size, output_size):
        return [
            torch.nn.BatchNorm1d(input_size),
            torch.nn.Linear(input_size, output_size),
            torch.nn.BatchNorm1d(output_size),
            torch.nn.Dropout(p=0.1)
        ]

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):

        sample = np.array(sample).reshape((1, self.input_dim))

        return super().predict(sample)