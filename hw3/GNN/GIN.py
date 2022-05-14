import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool


class GIN(nn.Module):
    def __init__(self,dataset,n_layers,d_hidden) -> None:
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(dataset.num_features, d_hidden), 
                nn.ReLU(), 
                nn.Linear(d_hidden,d_hidden),
                nn.ReLU(), 
                nn.BatchNorm1d(d_hidden)
            ),
            train_eps=True
        )
        self.convs = nn.ModuleList()
        for i in range(n_layers - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                    nn.Linear(d_hidden, d_hidden), 
                    nn.ReLU(), 
                    nn.Linear(d_hidden,d_hidden),
                    nn.ReLU(), 
                    nn.BatchNorm1d(d_hidden)
                 ),
                 train_eps=True
                )
            )
        self.fc1 = nn.Linear(d_hidden,d_hidden)
        self.fc2 = nn.Linear(d_hidden,dataset.num_classes)
    

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GINWithJK(nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super().__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(dataset.num_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=True))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.fc1 = nn.Linear(num_layers * hidden, hidden)
        else:
            self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

if __name__ == '__main__':
    dataset = 'Cora'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = GIN(dataset,n_layers=5,d_hidden=32).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters() , lr=0.01)


    for epoch in range(1, 201):
        train(data)
        train_acc, val_acc, test_acc = test(data)
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')