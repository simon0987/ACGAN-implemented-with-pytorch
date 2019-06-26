import torch
from torch.utils import data
from torchvision import transforms as T
import numpy as np
import PIL.Image

def pytorch_load_and_preprocess_image(path):
	    trans = []
	    trans.append(T.ToTensor())
	    trans.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

	    transform = T.Compose(trans)

	    img = PIL.Image.open(path)

	    return transform(img)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda")
class CartoonDataset(data.Dataset):
	def __init__(self,path,path_dict,label_dict,transforms):
		self.path = path
		self.path_dict = list(path_dict.values())
		self.label_dict = list(label_dict.values())
		self.transforms = transforms

	def __len__(self):
		return len(self.path_dict)

	def __getitem__(self,index):
		image_path = self.path_dict[index]
		img = pytorch_load_and_preprocess_image(image_path)
		label = torch.from_numpy(np.array(self.label_dict[index])).float().to(device)
		return img,label
