###
# Footfal
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Trainer
###

from timeit import default_timer as timer
import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,model, device, train_data, train_labels, val_data, val_labels, configs):
        self.device = device

        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.train_labels = train_labels.float()
        self.val_labels = val_labels.float()

        self.configs = configs

        self.optimizer = configs['trainer']['optimizer']
        self.criterion = configs['trainer']['criterion']
        self.epochs = configs['trainer']['epochs']

        self.set_training_config()

        logger.info(f"Train data: {type(train_data)}, {train_data.dtype}, {train_data.shape}")
        logger.info(f"Train labels: {type(train_labels)}, {train_labels.dtype}, {train_labels.shape}")

        logger.info(f"Train label summary: {torch.sum(train_labels, dim=0)}")  # Distribution of training labels
        logger.info(f"Validation label summary: {torch.sum(val_labels, dim=0)}")   # Distribution of validation labels
        logger.info(f"Input Mean: {train_data.mean()}, Std: {train_data.std()}")


    def set_training_config(self):
        print(f"Selected Optimizer = {self.optimizer}")
        if self.optimizer == "SGD":
            print(f"Setting opt = SGD")
            #print(self.model.parameters().shape)
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                              lr=self.configs['trainer']['learning_rate'],
                                              weight_decay=self.configs['trainer']['weight_decay'])
        else:
            raise NotImplementedError("Only SGD is supported for now")

        ## Loss Functions
        print(f"Selected Loss Function = {self.criterion}")
        if self.criterion == "MSE": 
            print(f"Loss function: Mean Squared Error, L2")
            self.criterion = nn.MSELoss()
            self.testCrit = nn.MSELoss()
        elif self.criterion == "MAE": 
            print(f"Loss function: Mean Absolute Error, L1")
            self.criterion = nn.L1Loss() 
            self.testCrit = nn.L1Loss()
        elif self.criterion == "Huber": # 
            print(f"Loss function: Huber")
            self.criterion = nn.HuberLoss()
            self.testCrit = nn.MSELoss()
        elif self.criterion == "Ridge":
                #rCost = self.rAlpha*self.model.weights.pow(2).sum()   # for Ridge Regression
            #self.criterion = self.ridgeLoss()
            self.rAlpha = self.alpha #this is ugly
            self.criterion = nn.MSELoss()
            self.testCrit = nn.MSELoss()
        elif self.criterion == "Lasso":
            self.lAlpha = self.alpha #this is ugly
            self.criterion = nn.MSELoss()
            self.testCrit = nn.MSELoss()
        elif self.criterion == "ENet":
            self.rAlpha = 1-self.r
            self.lAlpha = self.r
            self.criterion = nn.MSELoss()
            self.testCrit = nn.MSELoss()
        elif self.criterion == "Sigmoid":
            self.criterion = nn.BCEWithLogitsLoss()
            self.testCrit = nn.BCEWithLogitsLoss()
        elif self.criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
            self.testCrit = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Unsupported loss function")

    def train(self):
        self.model.train()
        lossArr = []
        accArr = []
        train_predsArr =[] # for confusion matrix
        
        self.train_data = self.train_data.to(self.device)
        self.train_labels = self.train_labels.to(self.device)

        numDatam = self.train_data.shape[0] 

        for epoch in range(self.epochs):
            nRun_epoch = 0
            correct_epoch = 0
            train_loss_epoch, train_acc_epoch = 0, 0

            for thisRun  in range(0, numDatam):
                nRun_epoch +=1
                #for thisRun  in range(0, n):
                #runStartTime = timer()

                #Batch, input ch, height, width
                data = self.train_data[thisRun].unsqueeze(0) # Batches of 1
                data = data.unsqueeze(0) # Batches of 1
                labels = self.train_labels[thisRun].unsqueeze(0)
                labels_logits = labels.argmax(dim=1) # for CrossEntropy loss

                #print(f"Data Shape: {data.shape}")
                #print(f"Data: {self.train_data}")
                self.optimizer.zero_grad()
    
                out_pred = self.model(data)

                #print("shape: {labels.shape}, dtype: {labels.dtype}")
                #print(f"Labels shape: {self.train_labels.shape}, dtype: {self.train_labels.dtype}")
                #print(f"shape: {out_pred.shape}, dtype: {out_pred.dtype}")

                loss = self.criterion(out_pred, labels_logits)

                loss.backward()
                self.optimizer.step()

                ## The Accuracy 
                out_pred_logits = torch.argmax(out_pred, 1) # Convert to logits
                #labels_logits = torch.argmax(labels,1) #convert to logits

                #labels_logits = torch.argmax(self.train_labels,1) #convert to logits
                #print(f"Training Predicted Shape: {out_pred_logits.shape}")
                #print(f"Training Predicted logits: {out_pred_logits}, labels: {labels_logits}")
                #print(f"Labels: {labels_logits}")

                correct_batch = out_pred_logits.eq(labels_logits).sum().item()
                correct_epoch += correct_batch
                train_loss_epoch += loss.item()

                print(f"epoch: {epoch}, run:{nRun_epoch}, Training Predicted : {out_pred.detach().cpu().numpy()}, labels: {labels_logits.detach().cpu().numpy()}, pred: {out_pred_logits.detach().cpu().numpy()}, correct: {100*correct_epoch/nRun_epoch}%")
                #print(f"Run correct: {thisTestCorr}, loss: {loss.item()}")
                #lossArr.append(loss.item())

                train_predsArr.append(out_pred) #for confusion matrix
                # Batch

            ## Epoch

            train_acc_epoch = 100 * correct_epoch / nRun_epoch
            train_loss_epoch = train_loss_epoch/nRun_epoch

            lossArr.append(train_loss_epoch)
            accArr.append(train_acc_epoch)
        
            #Timing
            #runEndTime = timer()
            #runTime = runEndTime - runStartTime
            #print(f"Correct: = {correct_epoch}, nRun: {nRun_epoch}")
        
            if epoch%1==0:
                print(f"Epoch: {epoch} | Train Loss: {train_loss_epoch:.3f} | Train Acc: {train_acc_epoch:.2f}")
                #print(f"Epoch: {epoch} | Train Loss: {thisTrainLoss:.3f} | Train Acc: {train_acc:.2f} | Elapsed Time: {runTime}")

        #trainPreds_np = np.array(train_predsArr)
        trainPreds_np = torch.stack(train_predsArr).detach().cpu().numpy()
        trainPreds_np_reshaped = trainPreds_np.reshape(trainPreds_np.shape[0], -1)

        np.savetxt("trainRes.csv", trainPreds_np_reshaped, delimiter=",", fmt="%.4f")
        #print(f"Final Training Predicted logits: {out_pred_logits}")
        #print(f"Labels: {labels_logits}")

        #print(f"Loss shape: {len(lossArr)}")
        thisFig, axis = plt.subplots(2, 1)
        axis[0].plot(range(len(lossArr)), lossArr)    
        axis[0].set_title("Training loss v Epoch")

        axis[1].plot(range(len(accArr)), accArr)    
        axis[1].set_title("Accuracy v Epoch")

        plt.savefig("output/trainingLoss.jpg")
        #plt.show()
        return 

    def validation(self):
        self.model.eval()
        test_loss, test_acc, total = 0, 0, 0
        correct,total = 0,0
        y_preds =[] # for confusion matrix
        y_logits =[] # for confusion matrix
        y_targs = []

        self.val_data = self.val_data.to(self.device)
        self.val_labels = self.val_labels.to(self.device)

        with torch.no_grad():
        #with torch.inference_mode():
            #Batch, input ch, height, width
            print(f"Test Data len: {self.val_data.shape}")
            print(f"Test Lables: {self.val_labels.shape}")

            nData = self.val_data.shape[0] 
            for thisRun  in range(0, nData):
                thisData = self.val_data[thisRun].unsqueeze(0) # Make into batch of 1
                thisData = thisData.unsqueeze(0) # make ch = 1
                thisLabel = self.val_labels[thisRun].unsqueeze(0) 
                #print(f"This Data len: {thisData.shape}")
                #print(f"This Lable: {self.val_labels[thisRun].shape}")

                val_pred = self.model(thisData)
                val_loss = self.criterion(val_pred, thisLabel)

                #print(f"This lab: {thisLabel}")
                val_pred_logits = torch.argmax(val_pred, 1) # Convert to logits
                labels_logits = torch.argmax(thisLabel,1) #convert to logits

                print(f"This pred: {val_pred}, lab: {labels_logits}")
                #print(f"This pred: {val_pred}, lab: {thisLabel}")


                correct += val_pred_logits.eq(labels_logits).sum().item()
                test_loss += val_loss.item()
                test_acc = 100 * correct / (thisRun+1)

                y_preds.append(val_pred) #for confusion matrix
                y_logits.append(val_pred_logits )
                y_targs.append(thisLabel) #for confusion matrix 

                print(f"Correct running: {correct}, loss: {test_loss}, accu: {test_acc}%")


            finalTestLoss = test_loss/nData

            #print(f"Final Val Predicted logits: {y_preds}")
            #print(f"Labels: {y_targs}")
            #print(f"y_logigs: {y_logits}")

            print(f"Test Loss: {finalTestLoss:.3f} | Test Acc: {test_acc:.2f}%")

        
        #from torchmetrics import ConfusionMatrix
        #from mlxtend.plotting import plot_confusion_matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn  as sns

        y_preds = torch.stack(y_preds).squeeze(1)
        y_targs = torch.stack(y_targs).squeeze(1)
        #print(f"pred: {y_preds}, targ: {y_targs}")

        # Move back to the CPU befor making numpy
        predicted_classes = torch.argmax(y_preds, dim=1).cpu().numpy()  # Shape: [13]
        true_classes = torch.argmax(y_targs, dim=1).cpu().numpy()  # Shape: [13]

        cm = confusion_matrix(true_classes, predicted_classes)
        print(f"Confusion Matrix:\n{cm}")

        #plt.figure()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig("output/validation.jpg")
        plt.show()
        
        return #test_loss, test_acc