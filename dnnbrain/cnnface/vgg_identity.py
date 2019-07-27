from cnnface.cnnface.model import vgg_face_dag as vf
import vgg_face as vf 
from torchvision import transforms
from torch import nn
import torch
import torch.utils.data as Data
import torchvision
import numpy as np


    #Create and save a new classifier (two class)
# model = vf.vgg_face('F:/Code/venv/cnnface/cnnface/model/vgg_face_dag.pth')
# for param in model.parameters():
#     param.requires_grad = False
# in_features = model.fc8.in_features
# out_features = 2
# new_fc8 = nn.Linear(in_features,out_features,bias = True)
# model.fc8 = new_fc8
#
# torch.save(model,'F:/Code/venv/cnnface/cnnface/model/vgg_identity_ori.pkl')


#Train the new model

# define the model,loss function,optimizer
vggI = torch.load('F:/Code/venv/cnnface/cnnface/model/vgg_identity_ori.pkl')

optimizer = torch.optim.Adam(vggI.parameters(),lr = 0.01)
loss_func = nn.CrossEntropyLoss()

#prepare data
images_path = ''
i,g,h,w = pd.read_''
Image.crop()
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()],
                               transforms.Normalize())

dataSet = torchvision.datasets.ImageFolder(images_path,transform)
train_loader = Data.DataLoader(dataSet,batch_size=8,shuffle=True)

#train model
for _,images,labels in train_loader:
    output = vggI(images)
    loss = loss_func(output,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
