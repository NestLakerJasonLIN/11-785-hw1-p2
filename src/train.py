import time
import torch
from tqdm.notebook import tqdm
from utils import *

def train_model(model, epochs, train_loader, eval_loader, criterion, optimizer, device,
                checkpoint, checkpoint_filename=""):
    assert_init_checkpoint(checkpoint) # TODO

    model.to(device)
    model.train()

    for epoch in range(checkpoint["model_statistics"]["curr_epoch"]+1,
                       checkpoint["model_statistics"]["curr_epoch"]+epochs+1):
        print("epoch: %d" % (epoch))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device=device)
        eval_loss, eval_acc = evaluate_model(model, eval_loader, criterion, device=device)
        checkpoint["model_statistics"]["train_losses"].append(train_loss)
        checkpoint["model_statistics"]["eval_losses"].append(eval_loss)
        checkpoint["model_statistics"]["eval_accs"].append(eval_acc)

        # update model statistics
        checkpoint["model_statistics"]["curr_epoch"] = epoch
        if eval_loss < checkpoint["model_statistics"]["best_eval_loss"]:
            checkpoint["model_statistics"]["best_eval_loss"] = eval_loss
            checkpoint["model_statistics"]["best_eval_acc"] = eval_acc
            checkpoint["model_statistics"]["best_eval_epoch"] = epoch
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            # save model as file exists
            if checkpoint_filename:
                torch.save(checkpoint, checkpoint_filename)
                print('The {}th epoch model is saved to {}'.format(epoch, checkpoint_filename))

        print_model_statistics(checkpoint)

        print('=' * 20)

    return checkpoint["model_statistics"]["train_losses"], \
           checkpoint["model_statistics"]["eval_losses"], \
           checkpoint["model_statistics"]["eval_accs"]

def train_epoch(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)
        predicted.detach_()
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    acc = (correct_predictions / total_predictions) * 100.0
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    print('Training Accuracy: ', acc, '%')
    return running_loss

def evaluate_model(model, test_loader, criterion, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
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