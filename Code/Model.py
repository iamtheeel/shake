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

#
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
    def __init__(self, numClasses:int, nCh:int, config=None):
        super().__init__() 
        '''
        MobileNet
        '''
        self.seed = config['trainer']['seed']
        self.isRegresh = config['model']['regression']
        if self.isRegresh:
            numOutputs = 1
        else:
            numOutputs = numClasses


        # Load the model from the zoo
        base_model = models.mobilenet_v2(weights=None)  # You can set `True` for pretrained weights

        #TODO: add a layer, or modify the first to change 2D to 3D
        #if timeDData:

        startFeature = 0 #Change to 1 to replace the first layer
        self.features  = base_model.features[startFeature:]

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # For classification
        lastLayerFeatureMap_size = 1280
        self.fc = nn.Linear(lastLayerFeatureMap_size, numOutputs)

    def forward(self, x: torch.Tensor):
        #TODO: 
        # run the new layers if timed
        # pass the first layer
        # Reshape the data

        # Run mobilenet
        x = self.features(x)

        #Clasifyer
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten before FC
        x = self.fc(x)  # Final classification

        return x


class leNetV5_timeDomain(nn.Module):
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
        #x = self.layer(x)
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
        linMult = 47792 # 1 = conv_3_out linConnections
        #linMult = 1
        stage1 = 512
        stage2 = 128
        self.linear = nn.Sequential( nn.Flatten(),
                                      nn.Linear(linMult*conv_3_out, stage1), #Output matrix (linMult*conv_3_out x stage1)
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