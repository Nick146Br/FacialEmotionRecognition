import os 
import cv2
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils.extract_landmarks import detect_landmark
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from utils.train_ import train
from utils.test_ import test
from tqdm import tqdm
from models.GNN import SSG
import torch
import random
import numpy as np

path_current = os.path.abspath(os.path.dirname(__file__))
path_dataset = os.path.join(path_current, 'dataset')


def get_class(name_path):
    if 'anger' in name_path: return 0
    if 'contempt' in name_path: return 1
    if 'disgust' in name_path: return 2
    if 'fear' in name_path: return 3
    if 'happy' in name_path: return 4
    if 'sadness' in name_path: return 5
    if 'surprise' in name_path: return 6

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
       
if __name__ == '__main__':
    seed = 42
    seed_everything(42)
    """HIPERPARAMETROS"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    learning_rate = 0.00001
    num_epochs = 25000
    in_channels = 2
    hidden_channels = 64
    num_hidden_layers = 4
    classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    out_channels = len(classes)
    """HIPERPARAMETROS"""
    
    get_all_ids = list()
    for classe in classes:
        path_classe = os.path.join(path_dataset, classe)
        ids = sorted(os.listdir(path_classe))
        for id_ in ids:
            if (id_.split('_')[0] in get_all_ids): continue
            get_all_ids.append(id_.split('_')[0])

    #train 80% test 20%
    train_ids, test_ids = train_test_split(get_all_ids, test_size=0.1, random_state=42)
    train_paths = list(); test_paths = list()

    for classe in classes:
        path_classe = os.path.join(path_dataset, classe)
        ids = os.listdir(path_classe)
        for id in ids:
            path_open = os.path.join(path_classe, id)
            if id.split('_')[0] in train_ids: train_paths.append(path_open)
            else: test_paths.append(path_open)

    data_st_train = list(); data_st_test = list()
    pbar = tqdm(total=len(train_paths), colour="white")
    pbar.set_description(f'Extracting Landmarks Train')
    for path in train_paths:
        classe = get_class(path)
        #save_dado
        path_save_np = path.replace('\\', '/')
        path_save_np = path_save_np.replace('dataset', 'dataset_landmarks')
        path_save_np = path_save_np.replace('png', 'pkl')
        path_aux = path_save_np.split('/')
        path_aux = path_aux[:-1]
        path_aux = '/'.join(path_aux)
        os.makedirs(path_aux, exist_ok=True)
        # if the file already exists, skip
        if os.path.exists(path_save_np):
            with open(path_save_np, 'rb') as f: 
                dado = pickle.load(f)
        else:
            dado = detect_landmark(path, classe)
            with open(path_save_np, 'wb') as f: pickle.dump(dado, f)
            
        data_st_train.append(dado)
        pbar.update(1)
        # break
        
    pbar.close()
    
    pbar = tqdm(total=len(test_paths), colour="white")
    pbar.set_description(f'Extracting Landmarks Test')    
    for path in test_paths:
        classe = get_class(path)
        path_save_np = path.replace('\\', '/')
        path_save_np = path_save_np.replace('dataset', 'dataset_landmarks')
        path_save_np = path_save_np.replace('png', 'pkl')
        path_aux = path_save_np.split('/')
        path_aux = path_aux[:-1]
        path_aux = '/'.join(path_aux)
        os.makedirs(path_aux, exist_ok=True)
        if os.path.exists(path_save_np):
            with open(path_save_np, 'rb') as f: 
                dado = pickle.load(f)
        else:
            dado = detect_landmark(path, classe)
            with open(path_save_np, 'wb') as f: pickle.dump(dado, f)
        data_st_test.append(dado)
        pbar.update(1)
        # break
    pbar.close()
    
    loader_train = DataLoader(data_st_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_test = DataLoader(data_st_test, batch_size=1)
    
    #in_channels, hidden_channels, num_hidden_layers, out_channels, device
    model = SSG(in_channels, hidden_channels, num_hidden_layers, out_channels, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_op = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train(model, optimizer, device, loader_train, 0, epoch, loss_op)
        test(model, optimizer, device, loader_test, 0, epoch, loss_op)

    save_path = os.path.join(path_current, 'model.pth')
    torch.save(model.state_dict(), save_path)