###
# Footfal
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Models
###
import torch
from torch import nn

#
    # Start with a supoer simple multi layer perseptron
class multilayerPerceptron(nn.Module):
    def __init__(self,input_features,num_classes,config):
        torch.manual_seed(1678)
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
    
class leNetV5(nn.Module):
    def __init__(self, numClasses: int, config):
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
        hidden_neurons = config['hidden_neurons']
        input_shape = 1

        self.conv2d_layers = [0,4,7]
        self.bn_layers = [1,5,8]
        self.shaLayEnd = 1
        self.midLayEnd = 2

        conv_1Lay = 12
        #conv_1Lay = 12
        conv_2Lay = 24
        torch.manual_seed(86)

        self.features = nn.Sequential(
                                        nn.Conv2d(in_channels=input_shape, out_channels=conv_1Lay, kernel_size=3, stride=1, padding=2),
                                        nn.BatchNorm2d(conv_1Lay),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        nn.Conv2d(in_channels=conv_1Lay, out_channels=conv_2Lay, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(conv_2Lay),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=conv_2Lay, out_channels=hidden_neurons, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(hidden_neurons),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                     )

        self.dropout = nn.Dropout(0.25)
        #linConnections = 32
        linConnections = 1
        #linMult = 605
        linMult = 309725 # 1 = 24x linConnections
        self.linear = nn.Sequential( nn.Flatten(),
                                      nn.Linear(linMult*conv_2Lay, linConnections), # conv_1Lay = 6, conv_2Lay = 16
                                      nn.ReLU(),
                                      nn.Linear(linConnections, linConnections), # conv_1Lay = 6, conv_2Lay = 16
                                      nn.ReLU()
                                      )  

        self.clasifyer = nn.Sequential(nn.Linear(linConnections, numClasses)  )


    def forward(self, x: torch.Tensor):
        #x = self.layer(x)
        x = self.features(x)
        #x = self.hiddenLayer(x)

        #x = self.dropout(x)
        x = self.linear(x)
        x = self.clasifyer(x)

        return x 