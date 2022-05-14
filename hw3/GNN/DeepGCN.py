import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DeepGCNLayer, GENConv



class DeepGCN(torch.nn.Module):
    def __init__(self,num_node_features, num_edge_features ,num_node_classes ,hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = nn.Linear(num_node_features, hidden_channels)
        if num_edge_features is not None:
            self.edge_encoder = nn.Linear(num_edge_features, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.5,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = nn.Linear(hidden_channels, num_node_classes)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)


def train(model,optimizer,data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(model,data):
    model.eval()
    out, accs = model(data.x, data.edge_index, data.edge_attr), []
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
    model = DeepGCN(dataset.num_features,None,dataset.num_classes,64,28).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters() , lr=0.01)


    for epoch in range(1, 201):
        train(model,optimizer,data)
        train_acc, val_acc, test_acc = test(model,data)
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')