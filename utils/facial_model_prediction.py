import torch
from ..models.GNN import SSG
import pdb
from torch_geometric.loader import DataLoader

class LoadModel():
    def __init__(self):
        self.in_channels = 2
        self.hidden_channels = 64
        self.num_hidden_layers = 4
        self.out_classes = 7
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = self.load_model()
        
    def load_model(self):
        model = SSG(self.in_channels, self.hidden_channels, self.num_hidden_layers, self.out_classes, self.device)
        model.to(self.device)
        model.load_state_dict(torch.load('./core/models/loaders/facial_emotion.pth'))
        model.eval()
        return model

    def predict(self, dado):
        # dado = detect_landmark(image, 'facial_emotion')
        data_st = [dado]
        loader_train = DataLoader(data_st, batch_size=1, shuffle=False)
        for dado in loader_train:
            dado = dado.to(self.device)
            x = dado.x
            edge_index = dado.edge_index
            edge_attr = dado.edge_attr
            pred = self.model_loaded(x, edge_index, edge_attr)
            pred = torch.softmax(pred, dim=1)
            #numpy
            pred = pred.cpu().detach().numpy()
        pred = pred*100
        pred = pred.round(2)
        print(pred)
        return pred
    