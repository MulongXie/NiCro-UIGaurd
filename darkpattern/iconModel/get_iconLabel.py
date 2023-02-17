import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import time
import os, json
from PIL import Image
import torch.multiprocessing as mp


def predict_label(images, model, class_names, transform_test, device):
    inputs = [transform_test(img) for img in images]
    inputs = torch.stack(inputs).to(device)

    # forward
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        outputs = nn.Softmax(dim=1)(outputs)
        values, preds = torch.max(outputs, 1)
        
    results = []
    for j in range(inputs.size()[0]):
        poss = values[j].item()
        if poss > 0.8:
            results.append([class_names[preds[j]], poss])
        else:
            results.append(["other", poss])
    return results

    # from sklearn.metrics import classification_report
    # y_true, y_pred = testing()
    # print(classification_report(y_true, y_pred, target_names=class_names))

    
