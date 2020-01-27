import torch
import numpy as np

def test_model(model, data_loader, device, save=False, filename="../data/test_pred.csv"):
    predicts = torch.LongTensor([])
    
    with torch.no_grad():
        model.eval()

        # no target in test dataset/data loader
        for batch_idx, data in enumerate(data_loader):   
            data = data.to(device)

            outputs = model(data)

            _, predict = torch.max(outputs.data, 1)
            
            predicts = torch.cat([predicts, predict])
    
    assert predicts.shape[0] == len(data_loader.dataset)
    
    if save:
        result = np.concatenate([np.arange(len(predicts)).reshape(-1, 1), predicts.numpy().reshape(-1, 1)], axis=1)
        np.savetxt(filename, result, fmt="%i", delimiter=",", header="id,label", comments="")
    
    return predicts