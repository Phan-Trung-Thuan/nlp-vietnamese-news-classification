import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

def calculate_loss_batch(model, X_batch, y_batch, criterion, device):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device) - 1 # Assuming labels are 1-indexed

    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    return loss

def calculate_acc_batch(model, X_batch, y_batch, device):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device) - 1

    y_pred = model(X_batch)
    y_pred = torch.argmax(y_pred, dim=1)
    acc = accuracy_score(y_pred.cpu(), y_batch.cpu())
    return acc

def eval_model(model, model_name, test_dataloader, device, save_path='.'):
    model = model.to(device)
    model.eval()
    
    y_pred = []
    y_truth = []
    
    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for X_batch, _, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device) - 1

            pred = model(X_batch)
            pred = torch.argmax(pred, dim=1)

            y_batch = y_batch.cpu()
            pred = pred.cpu()
            
            y_pred.append(pred)
            y_truth.append(y_batch)
        
    y_pred = torch.cat(y_pred)
    y_truth = torch.cat(y_truth)
    
    acc = accuracy_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred, average='weighted')
    cm = confusion_matrix(y_truth, y_pred)
    
    print(f'Accuracy: {acc}, F1: {f1}')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix of {model_name} model')
    
    plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix.png'))
    plt.show()
    return acc, f1

def train_model(model, model_name, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=20, save_path='weights'):
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    model = model.to(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    best_acc = 0
    best_ep = -1

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for X_batch, _, y_batch in train_dataloader:            
            optimizer.zero_grad()
            loss = calculate_loss_batch(model, X_batch, y_batch, criterion, device)
            loss.backward()
            
            if model_name in ('bi_lstm', 'bi_gru'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                
            optimizer.step()
            train_loss += loss.cpu().item()
            train_acc += calculate_acc_batch(model, X_batch, y_batch, device)

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        test_loss = 0.0
        test_acc = 0.0

        with torch.no_grad():
            for X_batch, _, y_batch in test_dataloader:
                loss = calculate_loss_batch(model, X_batch, y_batch, criterion, device)
                test_loss += loss.item()
                test_acc += calculate_acc_batch(model, X_batch, y_batch, device)
            
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print(f'Epoch {ep + 1}/{epochs}: Train loss {train_loss:.4f}, Test loss {test_loss:.4f}, Train acc {train_acc:.4f}, Test acc {test_acc:.4f}')

        if test_acc >= best_acc:
            best_acc = test_acc
            best_ep = ep
            torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}_best_loss_weight.pth'))
            print(f'Saved best model at epoch {ep + 1}')
            
    print(f'Best epoch at {best_ep + 1}, Best acc: {best_acc}')
            
    return {
        'train_loss': train_loss_list,
        'test_loss': test_loss_list,
        'train_acc': train_acc_list,
        'test_acc': test_acc_list
    }
