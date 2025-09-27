###
# Footfal
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Models
###
import torch #torch
from torch import nn
from torchvision import models
import math

import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Add a reShape for everybody to use
def reShapeTimeD(x, nCh, timePoints, target_height, target_width, target_size):
    ## TODO: OMG, clean up this mess!
    batchSize = x.shape[0]
    #logger.info(f"Batch Size: {batchSize}, outch: {nCh}, timePoints: {timePoints}")

    #logger.info(f"Before reshaping: {x.shape}, total elements: {x.numel()}")
    if target_width == 0:
        target_width = math.ceil(x.shape[2] / target_height)
        target_size = target_height * target_width
        #logger.info(f"Target Width: {target_width}")
    
    # Never trim, but Pad if necessary
    if timePoints != target_size:
        pad_size = target_size - timePoints
        #logger.info(f"Reshaping pad: {pad_size}")
        if pad_size > 0:
            #x = torch.cat((x, torch.zeros(batchSize, nCh, pad_size, device=x.device, dtype=x.dtype)), dim=2)  # Pad with zeros
            #x = torch.cat((x, torch.zeros(pad_size, dtype=x.dtype)))  # Pad with zeros
            x = nn.functional.pad(x, (0, max(0, pad_size)), "constant", 0)
            #logger.info(f"Reshaped: {x.numel()}")
        else:
            x = x[:target_size]  # Trim excess values
            #logger.error(f"neg pad!   {pad_size}")

    # Reshape to (batch, ch, height, width)
    #logger.info(f"before x.view: {x.shape}, total elements: {x.numel()}")
    x = x.view(batchSize, nCh, target_height, -1) #target_width)
    #x = x.view(batchSize, nCh, target_height, target_width)
    #logger.info(f"After reshaping: {x.shape}, total elements: {x.numel()}")

    return x


def replace_bn_with_gn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            num_groups = min(32, num_channels)  # Ensure num_groups does not exceed num_channels
            while num_channels % num_groups != 0 and num_groups > 1: # Ensure num_groups is a divisor of num_channels
                num_groups -= 1  # Reduce num_groups until it cleanly divides num_channels

            setattr(model, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            replace_bn_with_gn(module)

def add_dropout(model, p=0.3):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):  
            setattr(model, name, nn.Sequential(nn.Dropout(p=p), module))  # Ensure dropout is first
        elif isinstance(module, nn.Sequential):  # Check inside Sequential
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear):
                    setattr(module, sub_name, nn.Sequential(nn.Dropout(p=p), sub_module))
        else:
            add_dropout(module)

# Start with a supoer simple multi layer perseptron
class multilayerPerceptron(nn.Module):
    def __init__(self,input_features,num_classes,config):
        torch.manual_seed(config['trainer']['seed'])
        super().__init__() 
        hidden_neurons = config['hidden_neurons']
        self.dropout = nn.Dropout(0.2)
        self.inputLineLayer   = nn.Linear(input_features, hidden_neurons)
        self.batchNormaLayer = nn.BatchNorm1d(hidden_neurons)
        self.batchNormaOut = nn.BatchNorm1d(num_classes)
        self.outputLineLayer  = nn.Linear(hidden_neurons, num_classes)
        #self.activationLayer  = nn.Hardswish()
        self.activationLayer  = nn.LeakyReLU(negative_slope=0.01)
        self.outputActiLayer  = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = torch.flatten(x, start_dim=1)
        x = self.inputLineLayer(x)
        x = self.activationLayer(x)
        x = self.outputLineLayer(x)
        x = self.activationLayer(x)

        return x 
    
class MobileNet_v2(nn.Module):
    def __init__(self, numClasses:int, dataShape, folded=True, dropOut=0, config=None, timeD=False):
        super().__init__() 
        #TODO: Modify so its the same code for resnet
        '''
        MobileNet
        '''
        self.folded = folded

        self.seed = config['trainer']['seed']
        self.isRegresh = config['model']['regression']
        if self.isRegresh:
            numOutputs = 1
        else:
            numOutputs = numClasses

        # DataShape: [batch, colorCh, height, width]
        logger.info(f"Data Shape: {dataShape}")
        self.nCh = dataShape[1]

        # Load the model from zoo
        #base_model = models.regnet... so many to choose from
        base_model = models.mobilenet_v2(weights=None)  # You can set `True` for pretrained weights

        # Modify the first convolution layer to accept nCh channels instead of the default 3
        if folded: 
            #self.convForReshape = nn.Conv1d(in_channels=self.nCh, out_channels=32, kernel_size=5, stride=1, padding=2)
            self.convForReshape = nn.Conv1d(in_channels=self.nCh, out_channels=32, kernel_size=15, stride=2, padding=7)
            #self.convForReshape = nn.Conv1d(in_channels=self.nCh, out_channels=self.nCh, kernel_size=5, stride=1, padding=2)
            base_model.features[0][0] = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
            #base_model.features[0][0] = nn.Conv1d(in_channels=self.nCh, out_channels=self.nCh, kernel_size=5, stride=1, padding=2)
            self.timePoints = 1653
            self.target_width = 0
            self.target_height = 28 #41 #58 57
            #self.target_width = math.floor(math.sqrt(self.timePoints))
            #self.target_height = math.ceil(self.timePoints/self.target_width)
            self.target_size = self.target_width*self.target_height
            logger.info(f"Time Points: {self.timePoints}, Width: {self.target_width}, Height: {self.target_height}, new num points: {self.target_height*self.target_width}")
        else:
            self.timePoints = dataShape[2]
            base_model.features[0][0] = nn.Conv2d(self.nCh, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        #if(config['cwt']['doCWT']):
        self.timDData = timeD
        '''
        if(config['cwt']['wavelet']) != "None":
            # [Batch, Ch, Frequencies, TimePoints]
            logger.info(f"Wavelet Shape: {config['cwt']['wavelet']}")
            self.timDData = False
        else:
            # [Batch, Ch, TimePoints]
            logger.info(f"Time Domain Shape")
            self.timDData = True
            #the new count must be more than the number of timepoints

            #base_model.features[0][0] = nn.Conv1d(
            #                                      in_channels=self.nCh,
            #                                      out_channels=32,
            #                                      kernel_size=3,
            #                                      stride=2,
            #                                      padding=1,
            #                                      bias=False
            #                                     )
        '''


        startFeature = 0 #Change to 1 to replace the first layer instead of adding a new layer

        # Large batch and overfitting, re-check MobileNet with group norm, this was not done right?
        #replacing batch norm with group norm: a must for time d, unfolded
        replace_bn_with_gn(base_model)
        self.features  = base_model.features[startFeature:]

        self.global_pool = nn.AdaptiveAvgPool2d(1)


        lastLayerFeatureMap_size = 1280
        if dropOut > 0: # Add dropout layers to for overfitting
            #add_dropout(base_model, p=dropOut) #
            #self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential( nn.Dropout(p=dropOut),  # Explicit dropout before FC
                                     nn.Linear(lastLayerFeatureMap_size, numOutputs) )
        else:
            self.fc = nn.Linear(lastLayerFeatureMap_size, numOutputs)
            #print(self)



    def forward(self, x: torch.Tensor):
        # run the new layers if timed
        if self.timDData: 
            if self.folded == False:
                x = x.unsqueeze(-1) # Reshape the data to fit
            else:
                x = self.convForReshape(x) # Run it through a 1D conve first
                # Reshape the data
                x = reShapeTimeD(x, 32, self.timePoints, self.target_height, self.target_width, self.target_size)
                #x = reShapeTimeD(x, self.nCh, self.timePoints, self.target_height, self.target_width, self.target_size)


        # Run mobilenet
        x = self.features(x)

        #Clasifyer
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten before FC
        x = self.fc(x)  # Final classification

        return x

class leNetV5_timeDomain(nn.Module):
    def __init__(self, numClasses:int, dataShape, config):
    #def __init__(self, numClasses: int, nCh, config):
        super().__init__() 
        """
        LeNet-5:
            Convolution kernal = 5x5, stride=1, tanh
            Pooling, kernal = 2x2, stride 2, tanh

            Convolution kernal = 5x5, stride=1, tanh
            Pooling, kernal = 2x2, stride 2, tanh

            ## Fully connected
            Convolution kernal = 5x5, stride=1, tanh

            FC, tanh
            FC, softmax
        """
        logger.info(f"Init, dataShape: {dataShape}")

        nCh = 1 #dataShape[1]
        self.configsModel = config['model']['leNetV5']
        self.seed = config['trainer']['seed']

        self.conv2d_layers = [0,4,7]
        self.bn_layers = [1,5,8]
        self.shaLayEnd = 1
        self.midLayEnd = 2

        conv_1Lay = 12
        conv_2Lay = 12
        conv_3_out = 128
        #conv_2Lay = 24
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.features = nn.Sequential(
                                        nn.Conv2d(in_channels=nCh, out_channels=conv_1Lay, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(conv_1Lay),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        #nn.Conv2d(in_channels=conv_1Lay, out_channels=conv_2Lay, kernel_size=3, stride=1, padding=0),
                                        #nn.BatchNorm2d(conv_2Lay),
                                        #nn.ReLU(),

                                        nn.Conv2d(in_channels=conv_2Lay, out_channels=conv_3_out, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(conv_3_out),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                     )

        linMult = 825 # 1 = 24x linConnections
        stage1 = 512
        stage2 = 128
        self.linear = nn.Sequential( nn.Flatten(),
                                      nn.Linear(linMult*conv_3_out, stage1), 
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(stage1, stage2), 
                                      nn.ReLU()
                                      )  

        self.clasifyer = nn.Sequential(nn.Linear(128, numClasses)  )


    def forward(self, x: torch.Tensor):
        #data comes in as        (batch, ch, timepoints)
        x = x.data.unsqueeze(1) #(batch, 1, ch, timepoints)
        #logger.info(f"Data shape: {x.shape}")

        x = self.features(x)

        x = self.linear(x)
        x = self.clasifyer(x)

        return x 

class leNetV5_folded(nn.Module):
    def __init__(self, numClasses: int, dataShape, config):
        super().__init__() 
        """
        LeNet-5:
            Convolution kernal = 5x5, stride=1, tanh
            Pooling, kernal = 2x2, stride 2, tanh

            Convolution kernal = 5x5, stride=1, tanh
            Pooling, kernal = 2x2, stride 2, tanh

            ## Fully connected
            Convolution kernal = 5x5, stride=1, tanh

            FC, tanh
            FC, softmax
        """
        logger.info(f"Init, dataShape: {dataShape}")
        self.configsModel = config['model']['leNetV5']
        self.seed = config['trainer']['seed']

        self.timePoints = dataShape[2]
        self.target_height = config['model']['timeDImgHeight']
        #self.target_height = math.ceil(math.sqrt(self.timePoints))
        self.target_width = math.ceil(self.timePoints/self.target_height)

        self.batchSize = dataShape[0]
        self.nCh = dataShape[1]
        self.target_size = self.target_width*self.target_height

        logger.info(f"h: {self.target_height}, w: {self.target_width}, nPoints: {self.timePoints}, nCh: {self.nCh}")

        self.conv2d_layers = [0,4,7]
        self.bn_layers = [1,5,8]
        self.shaLayEnd = 1
        self.midLayEnd = 2

        conv_1Lay = 16
        conv_2Lay = 32
        conv_3_out = 64
        #conv_2Lay = 24
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        self.reshape_a = nn.Sequential(
                                    nn.Conv2d(in_channels=self.nCh, out_channels=conv_1Lay, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),  # (batch, conv_1Lay, 1, time_reduced)
                                    nn.BatchNorm2d(conv_1Lay),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Pooling only along time dimension
        )

        self.convForReshape = nn.Conv1d(in_channels=self.nCh, out_channels=self.nCh, kernel_size=5, stride=1, padding=2)


        self.features = nn.Sequential(
                                        nn.Conv2d(in_channels=self.nCh, out_channels=conv_1Lay, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(conv_1Lay),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        nn.Conv2d(in_channels=conv_1Lay, out_channels=conv_2Lay, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(conv_2Lay),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=conv_2Lay, out_channels=conv_3_out, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(conv_3_out),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                     )

        linMult = 154 #825 # 1 = 24x linConnections
        #stage1 = 64 #512
        stage2 = 64 #128
        self.linear = nn.Sequential( nn.Flatten(),
                                      nn.Linear(linMult*conv_3_out, conv_3_out),#stage1), 
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(conv_3_out, stage2), 
                                      nn.ReLU()
                                      )  

        self.clasifyer = nn.Sequential(nn.Linear(stage2, numClasses)  )

    def forward(self, x: torch.Tensor):
        x = self.convForReshape(x)

        #x = self.reShape(x)
        x = reShapeTimeD(x, self.nCh, self.timePoints, self.target_height, self.target_width, self.target_size)

        x = self.features(x)

        x = self.linear(x)
        x = self.clasifyer(x)

        return x 
    

class leNetV5_cwt(nn.Module):
    def __init__(self, numClasses: int, nCh, config):
        super().__init__() 
        """
        LeNet-5:
            Convolution kernal = 5x5, stride=1, tanh
            Pooling, kernal = 2x2, stride 2, tanh

            Convolution kernal = 5x5, stride=1, tanh
            Pooling, kernal = 2x2, stride 2, tanh

            ## Fully connected
            Convolution kernal = 5x5, stride=1, tanh

            FC, tanh
            FC, softmax
        """
        self.configsModel = config['model']['leNetV5']
        hidden_neurons = self.configsModel['hidden_neurons']
        self.seed = config['trainer']['seed']


        self.conv2d_layers = [0,4,7]
        self.bn_layers = [1,5,8]
        self.shaLayEnd = 1
        self.midLayEnd = 2

        conv_1_out = 8
        conv_2_out = 8
        conv_3_out = 8

        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.features = nn.Sequential(
                                        nn.Conv2d(in_channels=nCh, out_channels=conv_1_out, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(conv_1_out),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        nn.Conv2d(in_channels=conv_1_out, out_channels=conv_2_out, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(conv_2_out),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=conv_2_out, out_channels=conv_3_out, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(conv_3_out),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                     )

        #linMult = 486750 # 1 = conv_3_out linConnections
        # the number of cwt freques is tied up in this
        # The batch size also
        linMult = 263376 #11536 #47792 # 1 = conv_3_out linConnections
        #linMult = 1
        stage1 = 512
        stage2 = 128
        self.linear = nn.Sequential( nn.Flatten(),
                                      #nn.Linear(linMult*conv_3_out, stage1), #Output matrix (linMult*conv_3_out x stage1)
                                      nn.Linear(linMult, stage1), #Output matrix (linMult*conv_3_out x stage1)
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(stage1, stage2),
                                      nn.ReLU()
                                      )  

        self.clasifyer = nn.Sequential(nn.Linear(stage2, numClasses)  )


    def forward(self, x: torch.Tensor):
        #x = self.layer(x)
        x = self.features(x)

        x = self.linear(x)
        x = self.clasifyer(x)

        return x 


# Our own implementation of complex layers
# Average Pooling Layer
class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor):
        return torch.complex(
            self.pool(x.real),
            self.pool(x.imag)
        )
    
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, **bn_kwargs):
        super().__init__()
        # mirror BN params for real and imag
        self.bn_r = nn.BatchNorm2d(num_features, **bn_kwargs)
        self.bn_i = nn.BatchNorm2d(num_features, **bn_kwargs)

    def forward(self, x: torch.Tensor):
        # x: (N, C, H, W), complex64
        xr, xi = x.real, x.imag
        yr = self.bn_r(xr)
        yi = self.bn_i(xi)
        return torch.complex(yr, yi)
    

## complextorch requires: pip install depricated
##                        In the complextorch directory
##                        pip install  . --use-pep517
## Only using the activation layer from complextorch
import complextorch as cplx_torch
class leNet(nn.Module):
    def __init__(self, numClasses: int = 1, nCh: int = 3, complex: bool = False, config = None):
        super().__init__() 
        """
        LeNet-5:
            3x: Convolution --> Normalization --> Activation --> Pooling
            Flatten
            Linear (120) --> activation 
            Linear (84) --> activation 
            Linear (nClasses) --> 
            softmax

        """
        # Keep it deterministic
        self.seed = config['trainer']['seed']
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        fc1InputCount = 517280 
        self.complex = complex

        convolveLayer = nn.Conv2d
        if complex: 
            logger.info(" ***************  Using Complex Layers ***************  ")
            #convolveLayer = cplx_torch.nn.Conv2d 

            #batchNormLayer = cplx_torch.nn.BatchNorm2d  #RuntimeError: Expected both inputs to be Half, Float or Double tensors but got ComplexFloat and ComplexFloat
                                                         # RuntimeError: The size of tensor a (2129) must match the size of tensor b (6) at non-singleton dimension 4
            batchNormLayer = ComplexBatchNorm2d
            activation = lambda: cplx_torch.nn.CReLU(inplace=False) #Set to false if:one of the variables needed for gradient computation has been modified by an inplace operation:

            #poolLayer = cplx_torch.nn.AdaptiveAvgPool2d#cplx_pool.CAvgPool2d
            poolLayer = ComplexAvgPool2d  
            #linearLayer = cplx_torch.nn.Linear

            #The head is real valued for real valued loss functions
            self.fc1 = nn.Linear(2*fc1InputCount, 120) # We need to double the input size for R, I to real valued
        else:       
            logger.info(" ***************  Using Real Valued Layers ***************  ")

            batchNormLayer = nn.BatchNorm2d
            activation = lambda: nn.ReLU(inplace=True) # lambda to make it a function; Inplace to save memory (Set to false if:one of the variables needed for gradient computation has been modified by an inplace operation: )
            poolLayer = nn.AvgPool2d
            #linearLayer = nn.Linear

            self.fc1 = nn.Linear(fc1InputCount, 120)


        self.act = activation()
        self.pool = poolLayer(kernel_size=2, stride=2)

        dtype = torch.cfloat if complex else torch.float
        self.conv1 = convolveLayer(in_channels=nCh, out_channels=6, kernel_size=5, stride=1, padding=0, dtype=dtype)
        self.bn1 = batchNormLayer(6)
        self.conv2 = convolveLayer(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, dtype=dtype)
        self.bn2 = batchNormLayer(16)

        #TODO: add dropout layers?

        #The head is real valued for real valued loss functions
        #self.fc1 = linearLayer(fc1InputCount, 120)
        self.act_r = nn.ReLU(inplace=False) # The head is real valued for real valued loss functions
        self.fc2 = nn.Linear(120, 84)
        self.classifyer = nn.Linear(84, numClasses)

    def forward(self, x: torch.Tensor):
        #print(f"Input shape: {x.shape}, dtype: {x.dtype}")
        x = self.conv1(x)
        #print(f"After conv1 shape: {x.shape}, dtype: {x.dtype}")
        x = self.bn1(x) # Normalization Layer Here
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)# Normalization Layer Here
        x = self.act(x)
        x = self.pool(x)  
        x = torch.flatten(x, 1)
        #print(f"After flatten: {x.shape}, dtype: {x.dtype}")

        # Our loss functions are real valued, so we need to convert
        if self.complex:
            # The target is real valued, so we need to convert to real
            #x = torch.cat((x.real, x.imag), dim=1).flatten(1)  # (N, 2*C*H*W) float32 
            x = torch.view_as_real(x).flatten(1)  # (N, 2F) float32

        x = self.fc1(x) 
        #print(f"After fc1: {x.shape}, dtype: {x.dtype}")
        x = self.act_r(x)
        #print(f"After fc1, activation: {x.shape}, dtype: {x.dtype}")
        x = self.fc2(x)
        x = self.act_r(x)
        x = self.classifyer(x) 
        #print(f"After classification: {x.shape}, dtype: {x.dtype}")

        return x

class VGG(nn.Module):
    '''
        Very Deep Convolutional Networks for Large-Scale Image Recognition.
        https://arxiv.org/abs/1409.1556v6
        Karen Simonyan, Andrew Zisserman

        VGG model implementation based on: "vgg in pytorch" 
        https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py
    '''

    VGG_cfg = {"VGG11": [ 64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
               "VGG13": [ 64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
               "VGG16": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", ],
               "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M", ]
    }

    def make_layers(self, cfg, nCh=3 ):
        layers = []

        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            # Each non-pooling layer gets Conv2d, BatchNorm, ReLU
            layers += [nn.Conv2d(nCh, l, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(l)]
            layers += [nn.ReLU(inplace=True)]
            nCh = l
    
        return nn.Sequential(*layers)

    def __init__(self, numClasses: int = 1, nCh: int = 3, complex: bool = False, seed=86, cfg: str = "VGG16"):
        super().__init__() 
        # Keep it deterministic
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.features = self.make_layers(cfg=self.VGG_cfg[cfg], nCh=nCh)

        self.poolForClassifyer = nn.AdaptiveAvgPool2d((1, 1)) # Add a an adaptive pool to get a fixed size output to match the classifier input
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numClasses)
        )

        replace_bn_with_gn(self.features) # Replace all batch norm layers with group norm layers

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    
    def forward(self, x: torch.Tensor): # 4x downsampled: 9x256x2133
        #print(f"Initial shape: {x.shape}, dtype: {x.dtype}", flush=True)
        x = self.features(x) # output: 512x8x66
        #print(f"After Features shape: {x.shape}, dtype: {x.dtype}", flush=True)

        #x = x.view(x.size()[0], -1) instead of view, adaptive pool then flatten
        x = self.poolForClassifyer(x)        # output: (N, 512, 1, 1)
        #print(f"After pool shape: {x.shape}, dtype: {x.dtype}", flush=True)
        x = torch.flatten(x, 1)              # output: (N, 512)
        #print(f"After flatten shape: {x.shape}, dtype: {x.dtype}", flush=True)
        x = self.classifier(x) # oujtput: (N, numClasses)
        #print(f"After classifyer shape: {x.shape}, dtype: {x.dtype}", flush=True)

        return x