import torch
import torch.nn as nn
from torchvision import transforms
import json

from element_detection.classify_compo.CNN import CNN
from darkpattern.template_matching.template_matching import TemplateMatching


class modelLoader:
    def __init__(self):
        # *** UIED Compo Classifier ***
        self.model_compo_classifier = CNN()

        # *** UIGuard ***
        # Pytorch models
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # icon
        self.model_icon_path = r"darkpattern/iconModel/model_icon_clean_noisy81/best-0.93.pt"
        self.model_icon = None
        self.class_name_icon = json.load(open(r"darkpattern/iconModel/model_icon_clean_noisy81/iconModel_labels.json", "r"))
        # status
        self.model_status_path = r"darkpattern/statusModel/model_status_random3/99-train-0.99.pt"
        self.model_status = None
        self.class_name_status = ["checked", "unchecked", "other"]
        # template matching
        self.template_matcher = None

        self.template_matcher = TemplateMatching()

        # Pytorch models
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # icon
        self.model_icon_path = r"darkpattern/iconModel/model_icon_clean_noisy81/best-0.93.pt"
        self.model_icon = None
        self.class_name_icon = json.load(
            open(r"darkpattern/iconModel/model_icon_clean_noisy81/iconModel_labels.json", "r"))
        # status
        self.model_status_path = r"darkpattern/statusModel/model_status_random3/99-train-0.99.pt"
        self.model_status = None
        self.class_name_status = ["checked", "unchecked", "other"]

    def load_models(self):
        print('*** Load model for compo classifier ***')
        self.model_compo_classifier.load()
        print('*** Load Template Matcher ***')
        self.template_matcher = TemplateMatching()
        print('*** Load model for icon classifier ***')
        self.model_icon = torch.load(self.model_icon_path, map_location=self.device).to(self.device)
        self.model_icon.eval()
        self.model_icon.share_memory()
        print('*** Load model for status classifier ***')
        self.model_status = torch.load(self.model_status_path, map_location=self.device).to(self.device)
        self.model_status.eval()
        self.model_status.share_memory()

