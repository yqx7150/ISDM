from torch.utils.data import DataLoader, Subset,Dataset
import torchvision.transforms as transforms
import torchvision.transforms as T
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
class LoadDataset(Dataset):
    def __init__(self,file_list,transform=None):
        self.files = glob.glob(file_list+ '/*.png')
  
        
        
        self.transform = transform
        self.to_tensor = T.ToTensor()
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        siat_256_input = cv2.imread(self.files[index])
        siat_256_input = siat_256_input / 255.
        siat_256_input = siat_256_input.transpose(2, 0, 1)


        return siat_256_input