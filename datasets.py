import torch
from torch.utils.data import Dataset
import glob
import torchvision

class CLEVR(Dataset):
    def __init__(self,path_image,resolution=(112,112)) -> None:
        super().__init__()
        self.list_path = glob.glob(path_image+"\*.png")
        self.resize = torchvision.transforms.Resize(resolution)
    def __len__(self):
        return len(self.list_path)
    
    def __getitem__(self, index):
        image = torchvision.io.read_image(self.list_path[index])
        image = image[:-1,:,80:-80]
        image = self.resize(image)

        return image/255





if __name__=="__main__":
    data = CLEVR(r"C:\Users\Admin\Documents\Dataset\CLEVR_v1.0\CLEVR_v1.0\images\train")
    img = data.__getitem__(0)
    