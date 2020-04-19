import json
import os
import matplotlib.pyplot as plt
from detectron2.config import get_cfg, CfgNode

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines
        
def plot_loss_curve(cfg, model_name):
    experiment_metrics = load_json_arr(os.path.join(cfg.OUTPUT_DIR, 'metrics.json'))

    training_losses = []
    validation_losses = []
    idx = []

    for line in experiment_metrics:
        print(line)
        if 'total_loss' in line.keys() and 'validation_loss' in line.keys():
            idx.append(line['iteration'])
            training_losses.append(line['total_loss'])
            validation_losses.append(line['validation_loss'])

    plt.plot(idx, validation_losses, label="Validation Loss")
    plt.plot(idx, training_losses, label="Training Loss")
    plt.title('Loss curves for model ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Validation_Loss')
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'loss_curve.png'))