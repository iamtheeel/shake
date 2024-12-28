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
    def __init__(self,model, device, train_data_loader, val_data_loader,  configs):
    #def __init__(self,model, device, train_data, train_labels, val_data, val_labels, configs):
        self.device = device

        self.model = model.to(self.device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        #self.train_labels = train_labels.float()
        #self.val_labels = val_labels.float()

        self.configs = configs

        self.optimizer = configs['trainer']['optimizer']
        self.criterion = configs['trainer']['criterion']
        self.epochs = configs['trainer']['epochs']
        self.batchSize = configs['data']['batchSize']

        self.set_training_config()


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
        
        for epoch in range(self.epochs):
            nRun_epoch = 0
            correct_epoch = 0
            train_loss_epoch, train_acc_epoch = 0, 0
            epoch_StartTime = timer()

            for data, labels  in self.train_data_loader:
                data = data.to(self.device)
                #labels = labels.to(self.device)
                nRun_epoch +=1
                #for thisRun  in range(0, n):
                batch_StartTime = timer()

                #Batch, input ch, height, width
                #data = self.train_data[thisRun].unsqueeze(0) # Batches of 1
                #data = data.unsqueeze(0) # Batches of 1
                #labels = self.train_labels[thisRun].unsqueeze(0)
                labels_argMax = labels.argmax(dim=1) # for CrossEntropy loss
                labels_argMax = labels_argMax.to(self.device)

                #print(f"Data Shape: {data.shape}")
                #print(f"Data: {self.train_data}")
                self.optimizer.zero_grad()
    
                out_pred = self.model(data)

                #print("shape: {labels.shape}, dtype: {labels.dtype}")
                #print(f"Labels shape: {self.train_labels.shape}, dtype: {self.train_labels.dtype}")
                #print(f"shape: {out_pred.shape}, dtype: {out_pred.dtype}")

                loss = self.criterion(out_pred, labels_argMax)

                loss.backward()
                self.optimizer.step()

                ## The Accuracy 
                out_pred_argMax = torch.argmax(out_pred, 1) # Convert to argMax
                #labels_argMax = torch.argmax(labels,1) #convert to argMax

                #labels_argMax = torch.argmax(self.train_labels,1) #convert to argMax
                #print(f"Training Predicted Shape: {out_pred_argMax.shape}")
                #print(f"Training Predicted argMax: {out_pred_argMax}, labels: {labels_argMax}")
                #print(f"Labels: {labels_argMax}")

                correct_batch = out_pred_argMax.eq(labels_argMax).sum().item()
                correct_epoch += correct_batch
                train_loss_epoch += loss.item()
                batch_Time = timer() - batch_StartTime

                # Write a file
                info_str = f"epoch: {epoch}, batch:{nRun_epoch}"
                pred_str = f"Training Predicted : {out_pred.detach().cpu().numpy()}"
                lab_argMax_str = f"labels: {labels_argMax.detach().cpu().numpy()}"
                pred_argMax_str = f"pred: {out_pred_argMax.detach().cpu().numpy()}"
                percCorr_str = f"correct: {correct_batch}, {100*correct_batch/self.batchSize:.1f}%"
                batchTime_str = f"Batch Time: {batch_Time:.3f}s"
                print(f"{info_str} | {lab_argMax_str} | {pred_argMax_str} | {percCorr_str} | {batchTime_str}")
                #print(f"Run correct: {thisTestCorr}, loss: {loss.item()}")
                #lossArr.append(loss.item())

                train_predsArr.append(out_pred) #for confusion matrix
                # Batch

            ## Epoch
            batchSize = self.configs['data']['batchSize']
            train_acc_epoch = 100 * correct_epoch / (nRun_epoch*batchSize )
            train_loss_epoch = train_loss_epoch/nRun_epoch

            lossArr.append(train_loss_epoch)
            accArr.append(train_acc_epoch)
        
            #Timing
            epoch_runTime = timer() - epoch_StartTime
            #print(f"Correct: = {correct_epoch}, nRun: {nRun_epoch}")
        
            if epoch%1==0:
                print(f"Epoch: {epoch} | Train Loss: {train_loss_epoch:.3f} | Epoch Acc: {train_acc_epoch:.2f} | Epoch Time: {epoch_runTime:.3f}s")

        #trainPreds_np = np.array(train_predsArr)
        trainPreds_np = torch.stack(train_predsArr).detach().cpu().numpy()
        trainPreds_np_reshaped = trainPreds_np.reshape(trainPreds_np.shape[0], -1)

        np.savetxt("trainRes.csv", trainPreds_np_reshaped, delimiter=",", fmt="%.4f")
        #print(f"Final Training Predicted argMax: {out_pred_argMax}")
        #print(f"Labels: {labels_argMax}")

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

        with torch.no_grad():
        #with torch.inference_mode():
            nData = len(self.val_data_loader)
            print(f"Test Data len: {nData}")

            for data, labels  in self.val_data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                labels_argMax = torch.argmax(labels,1) #convert to argMax
                labels_argMax = labels_argMax.to(self.device)

                val_pred = self.model(data)
                val_loss = self.criterion(val_pred, labels_argMax)

                #print(f"This pred: {val_pred}, lab: {thisLabel}")
                val_pred_argMax = torch.argmax(val_pred, 1) # Convert to argMax
                labels_argMax = torch.argmax(labels,1) #convert to argMax

                correct += val_pred_argMax.eq(labels_argMax).sum().item()
                test_loss += val_loss.item()
                test_acc = 100 * correct / (nData+1)

                y_preds.append(val_pred) #for confusion matrix
                y_logits.append(val_pred_argMax )
                y_targs.append(labels) #for confusion matrix 

                # This gets written to a csv
                #print(f"This pred: {val_pred.detach().cpu().numpy()}, lab: {labels_argMax.detach().cpu().numpy()}")
                #print(f"Correct running: {correct}, loss: {test_loss/(thisRun+1):.3f}, accu: {test_acc:.1f}%")


            finalTestLoss = test_loss/nData

            #print(f"Final Val Predicted logits: {y_preds}")
            #print(f"Labels: {y_targs}")
            #print(f"y_logigs: {y_argMax}")

            print(f"Test Loss: {finalTestLoss:.3f} | Test Acc: {test_acc:.2f}%")

        
        #from torchmetrics import ConfusionMatrix
        #from mlxtend.plotting import plot_confusion_matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn  as sns

        y_preds = torch.stack(y_preds) # datums, batch, classes
        y_targs = torch.stack(y_targs)
        print(f"pred: {y_preds.shape}, targ: {y_targs.shape}")

        # Move back to the CPU befor making numpy
        predicted_classes = torch.argmax(y_preds, dim=2)  # Shape: [datums, batches]
        true_classes = torch.argmax(y_targs, dim=2) #

        pred_flat = predicted_classes.flatten().cpu().numpy() # Shape: [datums*batches]
        clas_flat = true_classes.flatten().cpu().numpy()

        cm = confusion_matrix(clas_flat, pred_flat)
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