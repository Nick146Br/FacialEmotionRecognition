from tqdm import tqdm
import torch
from sklearn import metrics
import numpy as np

def train(model, optimizer, device, train_loader, fold_teste, epoch, loss_op):
    
    model.train()
    total_loss = 0
    train_losses = list()
    train_pred = list()
    train_y = list()
    
    
    pbar = tqdm(total=len(train_loader.dataset), colour="red")
    pbar.set_description(f'Fold {fold_teste} - Epoch {epoch} - Training')
    
    for dado in train_loader:  
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y = dado.y.to(device)
            
            pred = model(dado.x.to(device), dado.edge_index.to(device), dado.edge_attr.to(device), dado.batch.to(device))
            loss = loss_op(pred, y)
            train_y.append(y.cpu().detach().numpy())
            train_pred.append(pred.softmax(1).argmax(1).cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        pbar.update(len(pred))

    pbar.close()
    
    train_pred = np.array(train_pred)
    train_y = np.array(train_y)
    train_pred = train_pred.flatten()
    train_y = train_y.flatten()
    acc_accuracy = metrics.accuracy_score(train_y, train_pred)
    
    train_losses = torch.Tensor(train_losses)
    train_losses = torch.mean(train_losses)
    print(f'Train: Acc -> {acc_accuracy} - Loss -> {train_losses}\n')
    
    return float(train_losses), acc_accuracy, train_pred, train_y