import numpy as np
import cv2
from PIL import Image
import pickle

import torch
import torchvision
from torchvision import transforms


def deeplabv3_remove_bg(img):
    img = np.array(img, dtype=np.uint8)
    # img = cv2.imread(image_path)
    # img = img[...,::-1] #BGR->RGB
    h,w,_ = img.shape
    # img = cv2.resize(img,(1000,1000))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('deeplabv3_resnet101.pkl', 'rb') as f:
        model = pickle.load(f)
    
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output = output.argmax(0)
    mask = output.byte().cpu().numpy()
    # mask = cv2.resize(mask,(w,h))
    # img = cv2.resize(img,(w,h))
    mask[mask>0] = 1.0 # NOTE: なぜか3が入っていたので
    mask = np.dstack([mask, mask, mask])
    img_masked = Image.fromarray(cv2.multiply(img, mask))
    index_masked = np.where(np.array(mask)[:,:,2]==0)
    return img_masked, index_masked
    
    