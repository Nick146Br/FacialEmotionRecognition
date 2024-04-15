from tqdm import tqdm
import torch
from sklearn import metrics

import numpy as np

def test(model, optimizer, device, train_loader, fold_teste, epoch, loss_op):
    
    model.eval()
    total_loss = 0
    test_losses = list()
    test_pred = list()
    test_y = list()
    
    pbar = tqdm(total=len(train_loader.dataset), colour="white")
    pbar.set_description(f'Fold {fold_teste} - Epoch {epoch} - Testing')
    
    for dado in train_loader:  
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():            
            y = dado.y.to(device)
            pred = model(dado.x.to(device), dado.edge_index.to(device), dado.edge_attr.to(device), dado.batch.to(device))
            loss = loss_op(pred, y)
            test_y.append(y.cpu().detach().numpy())
            test_pred.append(pred.softmax(1).argmax(1).cpu().detach().numpy())
            test_losses.append(loss.item())
            
        pbar.update(len(pred))

    pbar.close()
    test_losses = torch.Tensor(test_losses)
    test_losses = torch.mean(test_losses)
    
    test_pred = np.array(test_pred)
    test_y = np.array(test_y)
    test_pred = test_pred.flatten()
    test_y = test_y.flatten()
    acc_accuracy = metrics.accuracy_score(test_y, test_pred)
    
    print(f'Test: Acc -> {acc_accuracy} - Loss -> {test_losses}\n')
    return float(test_losses), acc_accuracy, test_pred, test_y