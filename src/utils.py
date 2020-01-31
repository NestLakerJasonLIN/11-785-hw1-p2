import torch

# ========== checkpoint specific ==============
def init_checkpoint(checkpoint):
    # model statistics
    checkpoint["model_statistics"] = {}
    checkpoint["model_statistics"]["best_eval_loss"] = float("inf")
    checkpoint["model_statistics"]["best_eval_acc"] = 0.0
    checkpoint["model_statistics"]["curr_epoch"] = 0
    checkpoint["model_statistics"]["best_eval_epoch"] = 0
    checkpoint["model_statistics"]["train_losses"] = []
    checkpoint["model_statistics"]["eval_losses"] = []
    checkpoint["model_statistics"]["eval_accs"] = []

    # model state dict
    checkpoint["model_state_dict"] = None
    checkpoint["optimizer_state_dict"] = None

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    assert_init_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def assert_init_checkpoint(checkpoint):
    assert checkpoint is not None
    for key, value in checkpoint["model_statistics"].items():
        assert value is not None

# =========== model specific ==================
def print_model_statistics(checkpoint):
    print("model statistics:")
    print("trained epochs: {} best eval epoch: {} best eval loss: {} best eval acc: {}".format(
        checkpoint["model_statistics"]["curr_epoch"], checkpoint["model_statistics"]["best_eval_epoch"],
        checkpoint["model_statistics"]["best_eval_loss"], checkpoint["model_statistics"]["best_eval_acc"]))