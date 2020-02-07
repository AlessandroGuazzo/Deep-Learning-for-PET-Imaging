###########################################################################################################################################



# THIS CODE CAN BE USED FOR THE TRAINING OF A LEARNED PRIMAL-DUAL ALGORITHM FOR miniPET IMAGE RECONSTRUCTION



# THE CODE IS USED TO TRAIN THE 3 ITERATIONS LPD ALGORITHM WITH miniPET DATA STARTING FROM THE 3 ITERATIONS ONE TRAINED ON SYNTHETHIC DATA



# THE TRAINING SET IS HYBRID WITH BOTH miniPET AND SYNTHETHIC DATA



###########################################################################################################################################


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
import nibabel
from torch.utils.data.dataset import Dataset, TensorDataset
from odl.contrib import fom 
import warnings
import glob, os
import GPUtil
import matplotlib.image as mpimg
import torch.nn.init as init
from torch.utils.data.dataset import Dataset, TensorDataset
import re


torch.manual_seed(123);  
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

# FUNCTIONS TO GENERATE SYNTHETIC DATA

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

ratio = 4 # CHANGE HERE miniPET SYNTHETIC RATIO

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

            return int(np.ceil(trainSino.shape[0]/ratio))

        else:

            return 0 

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

# FUNCTIONS TO PROCESS miniPET DATA
        
def volumeSplitter(inputTensor):

    
    volumes = inputTensor.numpy()
    newVolumes = np.zeros((volumes.shape[2],1,volumes.shape[3],volumes.shape[4]))

    i = 0
    while i < volumes.shape[0]:
            
        k = 0
        while  k < volumes.shape[2]:
            newVolumes[k+i*volumes.shape[2],0,:,:] = volumes[i,0,k,:,:]
            k = k+1
                
        i = i+1
            
    newVolumes = torch.tensor(newVolumes)
    return newVolumes

def GTprocessing(filename,uniform,top,bot):
    from scipy.ndimage import zoom
    gt_image_in=nibabel.load(filename)
    gt_image = gt_image_in.get_data()/1000 
    gt_image = np.transpose(gt_image, (0,2,1)) 
    gt_image = np.flip(gt_image,1)
    gt_image = zoom(gt_image,(1,0.395,0.395))
    temp = np.zeros((gt_image.shape[0],1,gt_image.shape[1],gt_image.shape[2]))
    temp[:,0,:,:]=gt_image
    gt_image = torch.tensor(temp)
    if uniform:
        gt_image[gt_image<bot]=0
        gt_image[np.logical_and(gt_image<top,gt_image>bot)]=torch.mean(gt_image[np.logical_and(gt_image<top,gt_image>bot)])
    return gt_image

def dataLoad(filename):
    scanner2 = Scanner() 
    scanner2.num_rings = 35 
    compression2 = Compression(scanner2) 
    compression2.max_diff_ring = 0
    compression2.max_num_segments = 0 
    compression2.num_of_views = 180 
    compression2.num_non_arccor_bins = 147 
    compression2.data_arc_corrected = True
    pet_projector2 = compression2.get_projector(restrict_to_cylindrical_FOV=False) 
    pet_projector_adj2=pet_projector2.adjoint
    real=nibabel.load(filename) 
    real_data = real.get_data() 
    real_data_ = np.transpose(real_data, (0,2,1)) 
    data = pet_projector2.range.element(real_data_) 
    data[data.asarray() < 0] = 0 
    a = np.zeros((1,1,data.shape[0],data.shape[1],data.shape[2]))
    a[0,0,:,:,:]=data
    data=torch.tensor(a)
    data=volumeSplitter(data)
    return data

# LOAD TRAIN SINOGRAMS AND GT OF MEASURES

print("Loading miniPET Train Data...")

first = True
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP1M2/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        if first:
            trainSino = dataLoad(file)
            first = False
            numSino = numSino+1
        else:
            trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
            numSino = numSino+1
              
for file in glob.glob("GTminiPETP1M2/*.mnc"):
    GT = GTprocessing(file,True,0.25,0.05)
    for i in range(numSino-1):
        GT = torch.cat((GT,GTprocessing(file,True,0.25,0.05)),dim=0)
           
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP1M1/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP1M1/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.25,0.07)),dim=0)
        
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP1M3/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP1M3/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.4,0.1)),dim=0)
        
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP1M4/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP1M4/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.5,0.15)),dim=0)
        
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP1M5/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1

for file in glob.glob("GTminiPETP1M5/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.65,0.1)),dim=0)

numSino = 0
for file in glob.glob("TrainSinogramsminiPETP1M6/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1

for file in glob.glob("GTminiPETP1M6/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.4,0.1)),dim=0)

numSino = 0
for file in glob.glob("TrainSinogramsminiPETP2M1/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP2M1/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.15,0.02)),dim=0)

numSino = 0
for file in glob.glob("TrainSinogramsminiPETP2M2/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP2M2/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.2,0.02)),dim=0)
        
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP2M4/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP2M4/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.3,0.05)),dim=0)
        
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP2M5/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP2M5/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.15,0.03)),dim=0)
        
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP2M6/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP2M6/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.15,0.03)),dim=0)
    
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP3M1/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP3M1/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.3,0.03)),dim=0)

numSino = 0
for file in glob.glob("TrainSinogramsminiPETP3M2/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP3M2/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.25,0.02)),dim=0)
        
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP3M3/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP3M3/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.1,0.02)),dim=0)
        
numSino = 0
for file in glob.glob("TrainSinogramsminiPETP3M4/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1
        
for file in glob.glob("GTminiPETP3M4/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.4,0.1)),dim=0)

numSino = 0
for file in glob.glob("TrainSinogramsminiPETP3M5/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1

for file in glob.glob("GTminiPETP3M5/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.4,0.1)),dim=0)


numSino = 0
for file in glob.glob("TrainSinogramsminiPETP3M6/*.sino.mnc"):
    if not re.match('Train.*x1',file):
        trainSino = torch.cat((trainSino,dataLoad(file)),dim=0)
        numSino = numSino+1

for file in glob.glob("GTminiPETP3M6/*.mnc"):
    for i in range(numSino):
        GT = torch.cat((GT,GTprocessing(file,True,0.15,0.05)),dim=0)

print(trainSino.shape)
print(GT.shape)
print("DONE!!!")

# LOAD SYNTHETIC DATA

print("Loading synthetic Data...")

dset_train=RandomEllipsoids(pet_projector.domain, num_ellipsoids=np.random.poisson(20), diag=200, train=True,transforms=trafo)

train_loader = torch.utils.data.DataLoader(dset_train, int(np.ceil(trainSino.shape[0]/ratio)), shuffle=True)

for i,im in enumerate(train_loader):
    if i == 1:
        break

train_images = im 

train_data = torch.tensor(np.zeros((train_images.shape[0],1,180,147)))

for i in range(train_images.shape[0]):
    noise = np.random.uniform(1/10,1/3)
    train_data[i,:,:,:] = generate_data(train_images[i:i+1,:,:,:],fwd_op_mod,noise_level=noise)

trainSino = torch.cat((trainSino,train_data),dim=0)
GT = torch.cat((GT,train_images),dim=0)

print(trainSino.shape)
print(GT.shape)
print("DONE!!!")

batch_size = 10 # CHANGE HERE BATCH SIZE
train_set = TensorDataset(GT,trainSino)
train_loader = torch.utils.data.DataLoader(train_set,batch_size ,shuffle=True)  

# MINIPET TEST DATA LOADING

print("Loading miniPET Test Data...")
first = True
numSino = 0
for file in glob.glob("Mouse4/*.sino.mnc"):
    if not re.match('Mou.*x1',file):
        if first:
            testSino = dataLoad(file)
            first = False
            numSino = numSino+1
        else:
            testSino = torch.cat((testSino,dataLoad(file)),dim=0)
            numSino = numSino+1
        
for file in glob.glob("Mouse4/GT.mnc"):
        GTtest = GTprocessing(file,False,0.4,0.1)
        for i in range(numSino-1):
            GTtest = torch.cat((GTtest,GTprocessing(file,False,0.4,0.1)),dim=0)

numSino = 0
for file in glob.glob("Mouse5/*.sino.mnc"):
    if not re.match('Mou.*x1',file):
        testSino = torch.cat((testSino,dataLoad(file)),dim=0)
        numSino = numSino+1

for file in glob.glob("Mouse5/GT.mnc"):
    for i in range(numSino):
        GTtest = torch.cat((GTtest,GTprocessing(file,False,0.4,0.1)),dim=0)

            
print(testSino.shape)
print(GTtest.shape)
print("DONE!!!")

test_set = TensorDataset(GTtest,testSino)
test_loader = torch.utils.data.DataLoader(test_set,batch_size ,shuffle=True) 


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


# LPD ALGORITHM DEFINITION

def LPD(Unet1,Unet2,Unet3,Unet4,Unet5,Unet6,sino,normalization,see):

        sino = sino.cuda().float()

        # Iter 0

        h0 = Unet1(sino)
        h0_img = fwd_op_adj_mod(h0)/normalization
        f0 = Unet2(h0_img)

        # Iter 1

        h0 = h0.cpu().detach().cuda()
        f0 = f0.cpu().detach().cuda()
        f0_data = fwd_op_mod(f0)
        outh1 = Unet3(torch.cat((h0,f0_data,sino),dim=1))
        h1 = outh1 + h0
        h1_img = fwd_op_adj_mod(h1)/normalization
        outf1 = Unet4(torch.cat((f0,h1_img),dim=1))
        f1 = outf1 + f0

        # Iter 2

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
    
    for batch_idx,(images,trainSino) in enumerate(train_loader):  
        
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
    for batch_idx,(test_im,testSino) in enumerate(test_loader):
        
    
        h2,f2 = LPD(Unet1,Unet2,Unet3,Unet4,Unet5,Unet6,testSino,normalisation,False)

        loss = loss_test(f2,test_im.cuda().float())
        losses.append(loss.item())
       
    loss_avg = np.mean(np.asarray(losses))
     
    print('\nTest set: Average loss: {:.6f}'.format(loss_avg)) 
    return loss.data

epochs = 10 # CHANGE HERE EPOCHS
log_interval = 500
loss_train = nn.SmoothL1Loss() 
loss_test = nn.SmoothL1Loss() 
learning_rate = 1.5e-3 
parameters = list(Unet1.parameters()) + list(Unet2.parameters()) + list(Unet3.parameters()) + list(Unet4.parameters())  + list(Unet5.parameters()) + list(Unet6.parameters()) 
optimizer = torch.optim.Adam(parameters, lr=learning_rate)
use_checkpoint = True


if use_checkpoint:
    
    try:
        checkpoint = torch.load('LearnedPrimalDual3IterationsEllipses.tar')
        Unet1.load_state_dict(checkpoint['model_state_dict1'])
        Unet2.load_state_dict(checkpoint['model_state_dict2'])
        Unet3.load_state_dict(checkpoint['model_state_dict3'])
        Unet4.load_state_dict(checkpoint['model_state_dict4'])
        Unet5.load_state_dict(checkpoint['model_state_dict5'])
        Unet6.load_state_dict(checkpoint['model_state_dict6'])
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

print("Training...")

train_error=[] 
test_error=[] 

for epoch in range(1, epochs + 1): 
    test_error.append(test())
    
    train_error.append(train(epoch)) 
    torch.save({
            'model_state_dict1': Unet1.state_dict(),
            'model_state_dict2': Unet2.state_dict(),
            'model_state_dict3': Unet3.state_dict(),
            'model_state_dict4': Unet4.state_dict(),
            'model_state_dict5': Unet5.state_dict(),
            'model_state_dict6': Unet6.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
                }, 'LPD3miniPEThybridLN'+str(epoch)+'ep.tar')
    print("Model Saved")
     
print("DONE!!!")