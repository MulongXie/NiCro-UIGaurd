import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import time
import os, json
from PIL import Image
import torch.multiprocessing as mp


def predict_status(images, model, class_names, transform_test, device):
    # print(type(images), type(images[0]))
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

        if poss > 0.9:
            results.append([class_names[preds[j]], poss])
        else:
            results.append(["other", poss])

        # target_folder = os.path.join("Test", results[-1][0])
        # if not os.path.exists(target_folder):
        #     os.makedirs(target_folder)
        # images[j].save(os.path.join(target_folder, str(j)+".jpg"))
    return results
