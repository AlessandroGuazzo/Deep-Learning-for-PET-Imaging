###########################################################################################################################################

# THIS CODE CONTINUES OR STARTS THE TRAINING OF A U-NET FOR PET IMAGE DENOISING 

# THE U-NET 3 LAYERS

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

numTrain = 25000 # CHANGE HERE TRAINING SET SIZE
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


# MLEM RECONSTRUCTION OPERATORS DEFINITIONS

class MLEM(odl.operator.Operator):
    def __init__(self, op, niter):
        super(MLEM, self).__init__(domain=pet_projector.range, range=pet_projector.domain, linear=True) 
        self.op = op 
        self.niter = niter 
    
    def _call(self, data):
        reco = self.range.one()
        odl.solvers.iterative.statistical.mlem(self.op, reco, data, niter=self.niter) 
        return reco

mlem_op_comp=MLEM(pet_projector,niter=10) # MLEM operator 10 iterations
mlem_op_comp_mod=OperatorAsModule(mlem_op_comp) 

mlem_op_net=MLEM(pet_projector,niter=1) # MLEM operator. 1 iteration
mlem_op_net_mod=OperatorAsModule(mlem_op_net) 

# GENERATE TRAINING AND TEST ELLIPSES DATASETS

trafo = transforms.Compose([transforms.ToTensor(),])

dset_train=RandomEllipsoids(pet_projector.domain, num_ellipsoids=np.random.poisson(20), diag=200, train=True,transforms=trafo)
dset_test=RandomEllipsoids(pet_projector.domain, num_ellipsoids=np.random.poisson(20), diag=200, train=False,transforms=trafo)


batch_size = 20 # CHANGE HERE batch size
train_loader = torch.utils.data.DataLoader(dset_train, numTrain, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dset_test, numTest, shuffle=True) 

print('Generating Train Dataset...')
for i,im in enumerate(train_loader):
    if i == 1:
        break

train_images = im 
train_data = torch.tensor(np.zeros((train_images.shape[0],1,180,147)))

for i in range(train_images.shape[0]):
    noise = np.random.uniform(1/10,1/3)
    train_data[i,:,:,:] = generate_data(train_images[i:i+1,:,:,:],fwd_op_mod,noise_level=noise) 

train_set = TensorDataset(train_images,mlem_op_net_mod(train_data))
train_loader = torch.utils.data.DataLoader(train_set,batch_size , shuffle=True)  
print('Done!!!')

print('Generating Test Dataset...')
for i,images in enumerate(test_loader):
    if i == 1:
        break

test_images = images 
test_data = torch.tensor(np.zeros((test_images.shape[0],1,180,147)))

for i in range(test_images.shape[0]):
    noise = np.random.uniform(1/10,1/3)
    test_data[i,:,:,:] = generate_data(test_images[i:i+1,:,:,:],fwd_op_mod,noise_level=noise)

reco = mlem_op_net_mod(test_data)
test_set = TensorDataset(test_images,reco)
test_loader = torch.utils.data.DataLoader(test_set,batch_size , shuffle=True) 
print('Done!!!')

# U-NET ARCHITECTURE DEFINITION

class Unet(nn.Module):
    
    def __init__(self):
        
        super(Unet,self).__init__()
        
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


Unet = Unet().cuda()

# MODEL SUMMARY

summary(Unet, input_size=(1, 147, 147))

# DEFINE TRAIN AND TEST FUNCTIONS 

def train(epoch): 
    
    Unet.train() 
    
    for batch_idx,(images,noisy_recos) in enumerate(train_loader): 
        
        optimizer.zero_grad() 
        output = Unet(noisy_recos.cuda()) 
        loss = loss_train(output, images.cuda().float()) 
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
    
    Unet.eval() 
    
    losses = []
    for batch_idx,(test_im,recons) in enumerate(test_loader):
        loss = loss_test(Unet(recons.cuda()), test_im.cuda().float())
        losses.append(loss.item())
       
    loss_avg = np.mean(np.asarray(losses))
     
    print('\nTest set: Average loss: {:.6f}'.format(loss_avg)) 
    
    return loss.data

# DEFINE TRAINING CHECKPOINT PARAMETERS

epochs = 25 # CHANGE HERE EPOCHS
log_interval=1000
loss_train = nn.SmoothL1Loss() 
loss_test = nn.SmoothL1Loss() 
learning_rate = 1.5e-3 

parameters = Unet.parameters()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# LOAD WHOLE MODEL CHECKPOINT TO RESUME TRAINING OR START FROM SCRATCH

use_checkpoint = False

if use_checkpoint:
    
    try:
        checkpoint = torch.load('previousCheckpoint.tar')
        Unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        Unet.to("cuda")

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
            'model_state_dict': Unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
                }, 'newCheckpoint.tar')
    print('Model Saved!!!')
