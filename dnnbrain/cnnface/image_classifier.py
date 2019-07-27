import torch
import torchvision
from PIL import Image
import torchvision as tv
from torchvision import transforms
import torch.nn.functional as F
from cnnface.cnnface.model import vgg_face_dag as vf
import numpy as np

#load model
model = vf.vgg_face_dag('F:/Code/venv/cnnface/cnnface/model/vgg_face_dag.pth')

# load and transform pictures
image = Image.open('F:/Code/venv/cnnface/testImage/0023_01.jpg')
image2 = torchvision.transforms.functional.crop(image,10.5,102.6,438.3,336.4)
image2.save('F:/Code/venv/cnnface/testImage/0023_01_testtransformcrop.jpg')
# transforms = transforms.Compose([transforms.functional.crop(image,73,73,221,184)]
#                                  # transforms.Resize((224, 224)),
#                                  # transforms.ToTensor()]
#                                 )


images = image.unsqueeze(0)

# image_classifier
for i in range(10):
    classifier_act = model(images)
    classifier_act = classifier_act.squeeze(0)
    classifier_num = classifier_act.detach().numpy()
    max_act_index = np.where(classifier_num==np.max(classifier_num))
    print(max_act_index[0] + 1)