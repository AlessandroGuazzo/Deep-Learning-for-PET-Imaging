###########################################################################################################################################

# THIS CODE CAN BE USED FOR THE TRAINING OF A LEARNED UPDATE WITH MEMORY ALGORITHM FOR PET IMAGE RECONSTRUCTION

# THE CODE IS USED TO TRAIN THE 4 ITERATION LUM ALGORITHM STARTING FROM THE 4 ITERATION LU PARAMETERS

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

numTrain = 125000 # CHANGE HERE TRAINING SET SIZE
numTest = 1000 # CHANGE HERE TEST SET SIZE

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
train_loader = torch.utils.data.DataLoader(dset_train, batch_size, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dset_test, batch_size, shuffle=True) 

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

# FIRST ITERATION ARCHITECTURE DEFINITION

class Unet1(nn.Module):
    
    def __init__(self):
        
        super(Unet1,self).__init__()
        
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),
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
                        nn.Conv2d(32,1,1,1,0),
                        nn.ReLU(inplace=True)
        )
        
          
    def forward(self,x):
        
        #print(x.shape)
        l1 = self.layer1(x)
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

Unet1 = Unet1().cuda()

# SECOND ITERATION ARCHITECTURE DEFINITION

class Unet2nd(nn.Module):
    
    def __init__(self):
        
        super(Unet2nd,self).__init__()
        
        self.layer12 = nn.Sequential(
                        nn.Conv2d(2,32,3,padding=1),
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
        l1 = self.layer12(x)
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

Unet2 = Unet2nd().cuda()

# OTHER ITERATIONS ARCHITECTURE DEFINITION

class Unet3rd(nn.Module):
    
    def __init__(self,n):
        
        super(Unet3rd,self).__init__()
        
        self.layer1 = nn.Sequential(
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
        l1 = self.layer1(x)
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

Unet3 = Unet3rd(3).cuda()
Unet4 = Unet3rd(4).cuda()

# INITIALIZE FIRST LAYER OF MEMORY NETS WITH XAVIER INITIALIZATION

Unet3.apply(weight_init)
Unet4.apply(weight_init)

# LEARNED UPDATE WITH MEMORY ALGORITHM DEFINITION

def LUM(Unet1,Unet2,Unet3,Unet4,sino,normalization,see):
        
        sino = sino.cuda().float()
        
        # ITER 1
        recos = fwd_op_adj_mod(sino)/normalization
        
        x0 = Unet1(recos)
        
        # ITER 2
        x0_data = fwd_op_mod(x0)
        
        data_diff0 = sino-x0_data
        
        diff_img0 = fwd_op_adj_mod(data_diff0)/normalization
        
        out0 = Unet2(torch.cat((x0,diff_img0),dim=1))
        
        x1 = out0 + x0
        
        x1 = x1.cpu().detach().cuda()
        
        # ITER 3
        x1_data = fwd_op_mod(x1)

        data_diff1 = sino-x1_data

        diff_img1 = fwd_op_adj_mod(data_diff1)/normalization
        
        out1 = Unet3(torch.cat((x0,x1,diff_img1),dim=1))

        x2 = out1 + x1
        
        x2 = x2.cpu().detach().cuda()
        
        # ITER 4
        x2_data = fwd_op_mod(x2)

        data_diff2 = sino-x2_data

        diff_img2 = fwd_op_adj_mod(data_diff2)/normalization
        
        out2 = Unet4(torch.cat((x0,x1,x2,diff_img2),dim=1))

        x3 = out2 + x2
        
        if see:
            num = 0
            
            plt.figure(figsize=(10,10))
            plt.imshow(sino[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('y0')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(recos[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('A*(y0)/||A||^2')
            plt.colorbar(fraction=0.046, pad=0.04)
        
            plt.figure(figsize=(10,10))
            plt.imshow(x0[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('x0')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(x0_data[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('x0_data')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(data_diff0[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('y0-x0_data')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(diff_img0[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('A*(y0-x0_data)/||A||^2')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(out0[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('out0')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(x1[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('x1')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(x1_data[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('x1_data')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(data_diff1[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('y0-x1_data')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(diff_img1[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('A*(y0-x1_data)/||A||^2')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(out1[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('out1')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(x2[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('x2')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(x2_data[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('x2_data')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(data_diff2[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('y0-x2_data')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(diff_img2[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('A*(y0-x2_data)/||A||^2')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(out2[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('out2')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.figure(figsize=(10,10))
            plt.imshow(x3[num,0,:,:].cpu().detach(),cmap='gray')
            plt.title('x3')
            plt.colorbar(fraction=0.046, pad=0.04)
        
        return x3

# DEFINE TRAIN AND TEST FUNCTIONS 

def train(epoch):
    
    Unet1.train() 
    Unet2.train()
    Unet3.train()
    Unet4.train()

    for batch_idx,images in enumerate(train_loader): 
        
        trainSino = torch.tensor(np.zeros((images.shape[0],1,180,147)))
        
        for i in range(images.shape[0]):
            noise = np.random.uniform(1/10,1/3)
            trainSino[i,:,:,:] = generate_data(images[i:i+1,:,:,:],fwd_op_mod,noise_level=noise) 
        
        optimizer.zero_grad() 
        
        x3 = LUM(Unet1,Unet2,Unet3,Unet4,trainSino,normalisation,False)

        loss = loss_train(x3, images.cuda().float()) 
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

    losses = []
    for batch_idx,test_im in enumerate(test_loader):
        
        testSino = torch.tensor(np.zeros((test_im.shape[0],1,180,147)))
        
        for i in range(test_im.shape[0]):
            noise = np.random.uniform(1/10,1/3)
            testSino[i,:,:,:] = generate_data(test_im[i:i+1,:,:,:],fwd_op_mod,noise_level=noise) 
        
        x3 = LUM(Unet1,Unet2,Unet3,Unet4,testSino,normalisation,False)

        loss = loss_test(x3, test_im.cuda().float())
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

parameters = list(Unet1.parameters()) + list(Unet2.parameters()) + list(Unet3.parameters()) + list(Unet4.parameters())
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# LOAD WHOLE MODEL CHECKPOINT TO RESUME TRAINING OR START FROM SCRATCH

use_checkpoint = True

if use_checkpoint:
    
    try:
        checkpoint = torch.load('LearnedUpdate4Iter.tar')
        Unet1.load_state_dict(checkpoint['model_state_dict1'])
        Unet2.load_state_dict(checkpoint['model_state_dict2'])
        Unet3.load_state_dict(checkpoint['model_state_dict3'],strict=False)
        Unet4.load_state_dict(checkpoint['model_state_dict4'],strict=False)
        Unet1.to("cuda")
        Unet2.to("cuda")
        Unet3.to("cuda")
        Unet4.to("cuda")
        
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
            'optimizer_state_dict': optimizer.state_dict(),
                }, 'LearnedUpdateMemory4Iter.tar')
    print('Model Saved!!!')
