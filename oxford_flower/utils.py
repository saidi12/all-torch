
import scipy.io
from torch.utils.data import DataLoader, Dataset, random_split
import scipy
from PIL import Image
import os
class OxfordDataset(Dataset):
    def __init__(self, rootdir, transform):
        self.rootdir = rootdir
        self.img_dir = os.path.join(rootdir, "jpg")
        
        labels_mat = scipy.io.loadmat(os.path.join(self.rootdir, "imagelabels.mat"))
        self.labels = labels_mat['labels'][0] - 1
        self.transform = transform
        self.error_logs = []
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        try:
            img_name = f'image_{idx + 1:05d}.jpg'

            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path)
            image.verify()
            image = Image.open(img_path)
            if image.size[0] < 32 or image.size[1] <32:
                raise ValueError(f"Image too small {image.size}")
            
            if image.mode != "RGB":
                image = image.convert('RGB')
            
            image = self.transform(image)
            label = self.labels[idx]
            
        
        

            return image, label
        except Exception as e:
            self.error_logs.append({
                'index':idx,
                'error':e,
                'path': img_path if 'img_path' in locals() else 'unknown'})
            next_idx = (idx+1) % len(self)
            return self.__getitem__(next_idx)

        

        
