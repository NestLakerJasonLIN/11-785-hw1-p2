import time
import torch

def train_model(model, epochs, train_loader, eval_loader, criterion, optimizer, device):
    train_losses = []
    eval_losses = []
    eval_accs = []

    model.to(device)

    for epoch in range(epochs):
        print("epoch: %d" % (epoch+1))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device=device)
        eval_loss, eval_acc = evaluate_model(model, eval_loader, criterion, device=device)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        print('='*20)
    
    return train_losses, eval_losses, eval_accs

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss

def evaluate_model(model, test_loader, criterion, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('evaluate Loss: ', running_loss)
        print('evaluate Accuracy: ', acc, '%')
        return running_loss, acc