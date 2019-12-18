###########################################################################################################################################

# THIS CODE CAN BE USED FOR THE TRAINING OF A LEARNED PRIMAL-DUAL ALGORITHM FOR PET IMAGE RECONSTRUCTION

# THE CODE IS USED TO TRAIN THE 3 ITERATIONS LPD ALGORITHM STARTING FROM THE 2 ITERATIONS ONE ACCORDING TO THE PROGRESSIVE LEARNING STRATEGY

# THE LEVEL OF NOISE IS EXTRACTED FROM AN UNIFORM DISTRIBUTION IN [1/10,1/3]

###########################################################################################################################################

# IMPORT NECESSARY PACKAGES:
import numpy as np 
import odlpet
import odl
import torch 
from torch import nn
from torch.nn import functional as F
from torch import optim 
import torchvision
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 
from odlpet.scanner.scanner import Scanner 
from odlpet.scanner.compression import Compression
from odl.contrib.torch import OperatorAsModule 
import time 
from odl.contrib import fom 
import warnings
import GPUtil
from torch.utils.data.dataset import Dataset, TensorDataset
from torchsummary import summary
import torch.nn.init as init

torch.manual_seed(123)
warnings.filterwarnings('ignore')
GPUtil.showUtilization()
torch.cuda.set_device(0) 
torch.cuda.empty_cache()
GPUtil.showUtilization()

# PROJECTOR DEFINITION

scanner = Scanner() 
scanner.num_rings = 1
compression = Compression(scanner) 
compression.max_num_segments = 0 
compression.num_of_views = 180 
compression.num_non_arccor_bins = 147 
compression.data_arc_corrected = True
pet_projector = compression.get_projector(restrict_to_cylindrical_FOV=False)
pet_projector_adj=pet_projector.adjoint
pet_projector.range
pet_projector.domain 
fwd_op_mod=OperatorAsModule(pet_projector)
fwd_op_adj_mod = OperatorAsModule(pet_projector_adj)
normalisation=(pet_projector.norm(estimate=True))**2
transformations = transforms.Compose([transforms.ToTensor()]) 

# DEFINITION OF FUNCTION NEEDED TO GENERATE THE TRAINING SET

def generate_ellipsoids_2d(space,num=np.random.poisson(20),diag=100):

    max_axis=diag**2 
    surround=odl.uniform_discr([0, 0], [int(np.ceil(np.sqrt(max_axis))),int(np.ceil(np.sqrt(max_axis)))],[int(np.ceil(np.sqrt(max_axis))), int(np.ceil(np.sqrt(max_axis)))], dtype='float32') 
    elipse_image=np.zeros((space.shape[1],space.shape[2]))

    for i in range(num):
        value=np.random.rand() 
        angle=np.random.rand()*2*np.pi 

        if np.sqrt(max_axis)>np.max(pet_projector.domain.shape): 
            x_center=2*np.random.rand()-1 
            y_center=2*np.random.rand()-1 
            axis1=np.random.exponential(scale=.5) 
            axis2=np.random.exponential(scale=.5) 

            if axis2>axis1:
                while axis2>axis1:
                    axis2=np.random.rand()

            elipse=[[value,axis1, axis2,x_center, y_center,angle]]

            el=odl.phantom.geometric.ellipsoid_phantom(surround, elipse).asarray()

            s=np.sum(el[int(np.sqrt(max_axis/2))-int(pet_projector.domain.shape[1]/2):int(np.sqrt(max_axis)/2)+int(pet_projector.domain.shape[1]/2),int(np.sqrt(max_axis)/2)-int(pet_projector.domain.shape[2]/2):int(np.sqrt(max_axis)/2)+int(pet_projector.domain.shape[2]/2)])
            if s>0: 
                elipse_image+=el[int(np.sqrt(max_axis)/2)-int(pet_projector.domain.shape[1]/2):int(np.sqrt(max_axis)/2)+int(pet_projector.domain.shape[1]/2)+1,int(np.sqrt(max_axis)/2)-int(pet_projector.domain.shape[2]/2):int(np.sqrt(max_axis)/2)+int(pet_projector.domain.shape[2]/2)+1]
        else:
            print('Please introduce a maximum value of the ellipse long axis larger than the size of the reconstruction space in both dimensions')
            break
        
    return elipse_image


def ellipse_batch(space, number_of_images,number_of_elipses): 
    
    images = np.zeros((number_of_images,space.shape[1],space.shape[2])) 
        
    for i in range(number_of_images):
        image = generate_ellipsoids_2d(space,number_of_elipses,200)
        images[i,:,:]=image
          
    return images 

numTrain = 10 # CHANGE HERE TRAINING SET SIZE
numTest = 10 # CHANGE HERE TEST SET SIZE

class RandomEllipsoids(Dataset):
    def __init__(self,space,num_ellipsoids,diag,train=True,transforms=None):
        """
        Args:
            space: image space where to obtain the dataset of ellipsoid images
            num: number of random ellipsoids
            diag: size of surrounding space used to compute ellipsoids outside the center of the FOV
            num_imag: number of images generated
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.space = pet_projector.domain 
        self.num_ellipsoids= np.random.poisson(15) 
        self.diag = diag 
        self.train=train 
        self.transforms = transformations 

    def __getitem__(self,index):
        data=torch.tensor(ellipse_batch(self.space,1,self.num_ellipsoids))
        
        return data

    def __len__(self):
        if self.train==True:
            return numTrain 
        else:
            return numTest 

def generate_data(images, operator, noise_level=1.):
    """Create noisy projection data from images.
    
    The data is generated according to ::
        
        data = Poisson(fwd_op(images)/10)
        
    where ``noise`` is standard white noise.
    
    Parameters
    ----------
    images : `Tensor`, shape ``(B, C, 28, 28)``
        Input images for the data generation.
        
    Returns
    -------
    data : `Tensor`, shape ``(B, C, 5, 41)``
        Projection data stack.
    """
    torch.manual_seed(123) 
    data = operator(images) 
    noisy_data = torch.tensor(np.random.poisson(data.cpu()/noise_level)*noise_level) 
    
    return noisy_data

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()]) 
    random_ellipsoids =         RandomEllipsoids(pet_projector.domain, num_ellipsoids=np.random.poisson(15), diag=100, train=True, transforms=transformations)

# GENERATE TRAINING AND TEST ELLIPSES DATASETS

trafo = transforms.Compose([transforms.ToTensor(),])

dset_train=RandomEllipsoids(pet_projector.domain, num_ellipsoids=np.random.poisson(20), diag=200, train=True,transforms=trafo)
dset_test=RandomEllipsoids(pet_projector.domain, num_ellipsoids=np.random.poisson(20), diag=200, train=False,transforms=trafo)


batch_size = 10 # CHANGE HERE batch size
train_loader = torch.utils.data.DataLoader(dset_train, numTrain, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dset_test, numTest, shuffle=True) 

# INITIALIZATION FUNCTION

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__ == '__main__':
    pass

# DUAL NETS ARCHITECTURE DEFINITION

class UnetD(nn.Module):
    
    def __init__(self,n):
        
        super(UnetD,self).__init__()
        
        self.layerc = nn.Sequential(
                        nn.Conv2d(n,32,3,padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32,32,3,stride = 1,padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True) 
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32,64,3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,3,stride = 1,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,128,3,stride = 1,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
        )
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer4 = nn.Sequential(
                        nn.Conv2d(128,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,256,3,stride = 1,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        
        )
        
        
        self.layerUP31 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,(0,1),(0,1)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
                        
        )
        
        self.layerUP32 = nn.Sequential(
                        nn.Conv2d(256,128,3,1,1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,128,3,1,1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
                        
        )
        
        self.layerUP41 = nn.Sequential(
                        nn.ConvTranspose2d(128,64,3,2,(1,0),(1,0)),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )
        
        self.layerUP42 = nn.Sequential(
                        nn.Conv2d(128,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )
        
        self.layerUP51 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,2,(1,0),(1,0)),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True)
        )
        self.layerUP52 = nn.Sequential(
                        nn.Conv2d(64,32,3,1,1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32,32,3,1,1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True)
        )
        
        self.layerUP61 = nn.Sequential(
                        nn.Conv2d(32,1,1,1,0)
        )
        
          
    def forward(self,x):
        
        #print(x.shape)
        l1 = self.layerc(x)
        #print(l1.shape)
        l2 = self.layer2(self.pool1(l1))
        #print(l2.shape)
        l3 = self.layer3(self.pool2(l2))
        #print(l3.shape)

        mid = self.layer4(self.pool3(l3))
        #print(mid.shape)
        
        l3up = self.layerUP31(mid)
        l3up = torch.cat((l3up,l3),dim=1)
        l3up = self.layerUP32(l3up)
        #print(l3up.shape)
        l4up = self.layerUP41(l3up)
        l4up = torch.cat((l4up,l2),dim=1)
        l4up = self.layerUP42(l4up)
        #print(l4up.shape)
        l5up = self.layerUP51(l4up)
        l5up = torch.cat((l5up,l1),dim=1)
        l5up = self.layerUP52(l5up)
        #print(l5up.shape)
        out = self.layerUP61(l5up)
        #print(out.shape)
        
        return out


Unet1 = UnetD(1).cuda()
Unet3 = UnetD(3).cuda()
Unet5 = UnetD(4).cuda()

# PRIMAL NETS ARCHITECTURE DEFINITION

class UnetP(nn.Module):
    
    def __init__(self,n):
        
        super(UnetP,self).__init__()
        
        self.layerc = nn.Sequential(
                        nn.Conv2d(n,32,3,padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32,32,3,stride = 1,padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True) 
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32,64,3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,3,stride = 1,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,128,3,stride = 1,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
        )
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer4 = nn.Sequential(
                        nn.Conv2d(128,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,256,3,stride = 1,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        
        )
        
        
        self.layerUP31 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
                        
        )
        
        self.layerUP32 = nn.Sequential(
                        nn.Conv2d(256,128,3,1,1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,128,3,1,1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
                        
        )
        
        self.layerUP41 = nn.Sequential(
                        nn.ConvTranspose2d(128,64,3,2,0,0),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )
        
        self.layerUP42 = nn.Sequential(
                        nn.Conv2d(128,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )
        
        self.layerUP51 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,2,0,0),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True)
        )
        self.layerUP52 = nn.Sequential(
                        nn.Conv2d(64,32,3,1,1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32,32,3,1,1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True)
        )
        
        self.layerUP61 = nn.Sequential(
                        nn.Conv2d(32,1,1,1,0)
        )
        
          
    def forward(self,x):
        
        #print(x.shape)
        l1 = self.layerc(x)
        #print(l12.shape)
        l2 = self.layer2(self.pool1(l1))
        #print(l2.shape)
        l3 = self.layer3(self.pool2(l2))
        #print(l3.shape)

        mid = self.layer4(self.pool3(l3))
        #print(mid.shape)
        
        l3up = self.layerUP31(mid)
        l3up = torch.cat((l3up,l3),dim=1)
        l3up = self.layerUP32(l3up)
        #print(l3up.shape)
        l4up = self.layerUP41(l3up)
        l4up = torch.cat((l4up,l2),dim=1)
        l4up = self.layerUP42(l4up)
        #print(l4up.shape)
        l5up = self.layerUP51(l4up)
        l5up = torch.cat((l5up,l1),dim=1)
        l5up = self.layerUP52(l5up)
        #print(l5up.shape)
        out = self.layerUP61(l5up)
        #print(out.shape)
        
        return out

Unet2 = UnetP(1).cuda()
Unet4 = UnetP(2).cuda()
Unet6 = UnetP(3).cuda()

# INITIALIZE FIRST LAYER OF ALL NETS WITH XAVIER INITIALIZATION

Unet1.apply(weight_init)
Unet2.apply(weight_init)
Unet3.apply(weight_init)
Unet4.apply(weight_init)
Unet5.apply(weight_init)
Unet6.apply(weight_init)

# LEARNED PRIMAL DUAL ALGORITHM DEFINITION

def LPD(Unet1,Unet2,Unet3,Unet4,Unet5,Unet6,sino,normalization,see):
        
        sino = sino.cuda().float()
        
        # Iter 1
        h0 = Unet1(sino)
        
        h0_img = fwd_op_adj_mod(h0)/normalization
        
        f0 = Unet2(h0_img)
        
        # Iter 2
        h0 = h0.cpu().detach().cuda()
        f0 = f0.cpu().detach().cuda()
        
        f0_data = fwd_op_mod(f0)
        
        outh1 = Unet3(torch.cat((h0,f0_data,sino),dim=1))
        
        h1 = outh1 + h0
        
        h1_img = fwd_op_adj_mod(h1)/normalization
        
        outf1 = Unet4(torch.cat((f0,h1_img),dim=1))
        
        f1 = outf1 + f0

        # Iter 3
        h1 = h1.cpu().detach().cuda()
        f1 = f1.cpu().detach().cuda()

        f1_data = fwd_op_mod(f1)

        outh2 = Unet5(torch.cat((h1,h0,f1_data,sino),dim=1))  

        h2 = outh2 + h1

        h2_img = fwd_op_adj_mod(h2)/normalization

        outf2 = Unet6(torch.cat((f1,f0,h2_img),dim=1))

        f2 = outf2 + f1

        if see:
            num = 0
            plt.figure(figsize=(10,10))
            plt.imshow(sino[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('g')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(h0[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('h0')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(h0_img[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('h0_img')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(f0[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('f0')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(f0_data[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('f0_data')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(outh1[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('outh1')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(h1[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('h1')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(h1_img[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('h1_img')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(outf1[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('outf1')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(f1[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('f1')
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.figure(figsize=(10,10))
            plt.imshow(f1_data[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('f1_data')
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.figure(figsize=(10,10))
            plt.imshow(outh2[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('outh2')
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.figure(figsize=(10,10))
            plt.imshow(h2[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('h2')
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.figure(figsize=(10,10))
            plt.imshow(h2_img[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('h2_img')
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.figure(figsize=(10,10))
            plt.imshow(outf2[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('outf2')
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.figure(figsize=(10,10))
            plt.imshow(f2[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('f2')
            plt.colorbar(fraction=0.046, pad=0.04)

        return h2 ,f2

# DEFINE TRAIN AND TEST FUNCTIONS 

def train(epoch): 
    
    Unet1.train() 
    Unet2.train()
    Unet3.train() 
    Unet4.train()
    Unet5.train() 
    Unet6.train()

    for batch_idx,images in enumerate(train_loader):   
        
        trainSino = torch.tensor(np.zeros((images.shape[0],1,180,147)))
        
        for i in range(images.shape[0]):
            noise = np.random.uniform(1/10,1/3)
            trainSino[i,:,:,:] = generate_data(images[i:i+1,:,:,:],fwd_op_mod,noise_level=noise) 
            
        optimizer.zero_grad()
        
        h2,f2 = LPD(Unet1,Unet2,Unet3,Unet4,Unet5,Unet6,trainSino,normalisation,False)

        loss = loss_train(f2, images.cuda().float()) 
        loss.backward() 
        optimizer.step() 
        
        if batch_idx % log_interval == 0: 
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'
                  ''.format(epoch, batch_idx * len(images),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            loss))
        if batch_idx==len(train_loader)-1:
            return loss.data

def test(): 
    
    Unet1.eval() 
    Unet2.eval()
    Unet3.eval() 
    Unet4.eval()
    Unet5.eval() 
    Unet6.eval()
    
    losses = []
    for batch_idx,test_im in enumerate(test_loader):
        
        testSino = torch.tensor(np.zeros((test_im.shape[0],1,180,147)))
        
        for i in range(test_im.shape[0]):
            noise = np.random.uniform(1/10,1/3)
            testSino[i,:,:,:] = generate_data(test_im[i:i+1,:,:,:],fwd_op_mod,noise_level=noise) 
        
        h2,f2 = LPD(Unet1,Unet2,Unet3,Unet4,Unet5,Unet6,testSino,normalisation,False)

        loss = loss_test(f2,test_im.cuda().float())
        losses.append(loss.item())
       
    loss_avg = np.mean(np.asarray(losses))
     
    print('\nTest set: Average loss: {:.6f}'.format(loss_avg)) 
    return loss.data

# DEFINE TRAINING CHECKPOINT PARAMETERS

epochs = 3 # CHANGE HERE EPOCHS
log_interval = 1000
loss_train = nn.SmoothL1Loss() 
loss_test = nn.SmoothL1Loss() 
learning_rate = 1.5e-3 

parameters = list(Unet1.parameters()) + list(Unet2.parameters()) + list(Unet3.parameters()) + list(Unet4.parameters())  + list(Unet5.parameters()) + list(Unet6.parameters())
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# LOAD WHOLE MODEL CHECKPOINT TO RESUME TRAINING OR START FROM SCRATCH

use_checkpoint = True


if use_checkpoint:
    
    try:
        checkpoint = torch.load('LearnedPrimalDual2Iter.tar')
        Unet1.load_state_dict(checkpoint['model_state_dict1'],strict=False) # CHANGE NAME OF 1ST LAYER SO IT WON'T BE LOADED WITH STRICT
        Unet2.load_state_dict(checkpoint['model_state_dict2'],strict=False)
        Unet3.load_state_dict(checkpoint['model_state_dict3'],strict=False)
        Unet4.load_state_dict(checkpoint['model_state_dict4'],strict=False)
        Unet5.load_state_dict(checkpoint['model_state_dict3'],strict=False)
        Unet6.load_state_dict(checkpoint['model_state_dict4'],strict=False)
        Unet1.to("cuda")
        Unet2.to("cuda")
        Unet3.to("cuda")
        Unet4.to("cuda")
        Unet5.to("cuda")
        Unet6.to("cuda")

        print("\n--------model restored--------\n")
    except:
        print("\n--------model NOT restored--------\n")
        pass
else:
    print("\n--------model NOT restored--------\n")


GPUtil.showUtilization()

# TRAINING

print('Training...')

train_error=[] 
test_error=[] 

for epoch in range(1, epochs + 1): 
    train_error.append(train(epoch)) 
    test_error.append(test()) 
    
print('Done!!!')

# SAVE MODEL PARAMETERS AND OPTIMIZER STATUS

save = False

if save:
    torch.save({
            'model_state_dict1': Unet1.state_dict(),
            'model_state_dict2': Unet2.state_dict(),
            'model_state_dict3': Unet3.state_dict(),
            'model_state_dict4': Unet4.state_dict(),
            'model_state_dict5': Unet5.state_dict(),
            'model_state_dict6': Unet6.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
                }, 'LearnedPrimalDual3Iter.tar')
    print('Model Saved!!!')
