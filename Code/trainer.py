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
import csv

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,model, device, train_data_loader, val_data_loader,  configs, logFile, dateTime_str):
    #def __init__(self,model, device, train_data, train_labels, val_data, val_labels, configs):
        self.device = device

        self.model = model.to(self.device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.configs = configs
        self.outputDir = configs['outputDir']



        self.optimizer = configs['trainer']['optimizer']
        self.criterion = configs['trainer']['criterion']
        self.epochs = configs['trainer']['epochs']
        self.batchSize = configs['data']['batchSize']
        modelName = configs['model']['name']


        self.hyperPeramStr = f"{modelName}, batch size:{self.batchSize}, epochs:{self.epochs}"

        self.dateTime_str = dateTime_str
        self.logfile = logFile

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

        fieldnames = ['epoch', 'batch', 'batch correct', 'accuracy (%)', 'loss', 'time(s)']
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
            writer.writeheader()
        
        for epoch in range(self.epochs):
            batchNumber = 0
            correct_epoch = 0
            train_loss_epoch, train_acc_epoch = 0, 0
            epoch_StartTime = timer()

            for data, labels  in self.train_data_loader:
                data = data.to(self.device)
                #labels = labels.to(self.device)
                batchNumber +=1
                #for thisRun  in range(0, n):
                batch_StartTime = timer()

                #Batch, input ch, height, width
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
                '''
                info_str = f"epoch: {epoch}, {batchNumber}"
                pred_str = f"Training Predicted : {out_pred.detach().cpu().numpy()}"
                lab_argMax_str = f"labels: {labels_argMax.detach().cpu().numpy()}"
                pred_argMax_str = f"pred: {out_pred_argMax.detach().cpu().numpy()}"
                percCorr_str = f"correct: {correct_batch}, {100*correct_batch/self.batchSize:.1f}%"
                batchTime_str = f"Batch Time: {batch_Time:.3f}s"
                print(f"{info_str} | {lab_argMax_str} | {pred_argMax_str} | {percCorr_str} | {batchTime_str}")
                '''
                with open(self.logfile, 'a', newline='') as csvFile:
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
                    writer.writerow({'epoch'            : epoch,
                                     'batch'            : batchNumber,
                                     'batch correct'    : correct_batch, 
                                     'accuracy (%)'     : 100*correct_batch/self.batchSize, 
                                     'loss'             : loss.item(),
                                     'time(s)'          : batch_Time
                                     })

                #print(f"Run correct: {thisTestCorr}, loss: {loss.item()}")
                #lossArr.append(loss.item())

                train_predsArr.append(out_pred) #for confusion matrix
                #End  Batch

            ## Epoch
            batchSize = self.configs['data']['batchSize']
            train_acc_epoch = 100 * correct_epoch / (batchNumber*batchSize )
            train_loss_epoch = train_loss_epoch/batchNumber

            lossArr.append(train_loss_epoch)
            accArr.append(train_acc_epoch)
        
            #Timing
            epoch_runTime = timer() - epoch_StartTime
            #print(f"Correct: = {correct_epoch}, nRun: {batchNumber}")
        
            if epoch%1==0:
                print(f"Epoch: {epoch} | Train Loss: {train_loss_epoch:.3f} | Epoch Acc: {train_acc_epoch:.2f} | Epoch Time: {epoch_runTime:.3f}s")

            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
                writer.writerow({'epoch'            : epoch,
                                 'accuracy (%)'     : train_acc_epoch, 
                                 'loss'             : train_loss_epoch,
                                 'time(s)'             : epoch_runTime
                                 })
        # End Epoch

        # write the data 
        #trainPreds_np = torch.stack(train_predsArr).detach().cpu().numpy()
        #trainPreds_np_reshaped = trainPreds_np.reshape(trainPreds_np.shape[0], -1)

        #np.savetxt("trainRes.csv", trainPreds_np_reshaped, delimiter=",", fmt="%.4f")
        #print(f"Final Training Predicted argMax: {out_pred_argMax}")
        #print(f"Labels: {labels_argMax}")

        #print(f"Loss shape: {len(lossArr)}")
        fig, axis = plt.subplots(2, 1)
        fig.subplots_adjust(top = 0.92, hspace = .05, left= 0.125, right = 0.99)
        axis[0].plot(range(len(lossArr)), lossArr)    
        axis[0].set_title(f"{self.hyperPeramStr}")
        axis[0].set_ylabel("Training Loss")
        axis[0].get_xaxis().set_visible(False)

        axis[1].plot(range(len(accArr)), accArr)    
        axis[1].set_ylabel("Accuracy (%)")
        axis[1].get_xaxis().set_visible(True)
        axis[1].set_xlabel("Epoch")

        plt.savefig(f"output/{self.dateTime_str}_trainingLoss.jpg")
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


            finalValLoss = test_loss/nData

            #print(f"Final Val Predicted logits: {y_preds}")
            #print(f"Labels: {y_targs}")
            #print(f"y_logigs: {y_argMax}")

            print(f"Validation Loss: {finalValLoss:.3f} | Test Acc: {test_acc:.2f}%")
            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile, dialect='unix')
                writer.writerow(["-------", "Validation"])
                writer.writerow(["Loss", "Accuracy"])
                writer.writerow([finalValLoss, test_acc])

        
        #from torchmetrics import ConfusionMatrix
        #from mlxtend.plotting import plot_confusion_matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn  as sns

        # Convert to torch tensor
        y_preds = torch.stack(y_preds) # datums, batch, classes
        y_targs = torch.stack(y_targs)

        classes = self.configs['data']['classes']
        if(self.configs['debugs']['writeData']):
            y_preds_flt = y_preds.view(-1,3)
            y_targs_flt = y_targs.view(-1,3)
            y_preds_targets = torch.cat((y_preds_flt, y_targs_flt), dim=1)
            print(f"pred: {y_preds_flt.shape}, targ: {y_targs_flt.shape}, combined: {y_preds_targets.shape}")
            with open(f"{self.configs['outputDir']}/{self.dateTime_str}_results.csv", 'w', newline='') as csvFile:
                writer = csv.writer(csvFile, dialect='unix')
                writer.writerow(['Predictions', '', '', 'Labels'])
                writer.writerow(classes + classes)
                for row in y_preds_targets:
                    writer.writerow(row.tolist())

        # Move back to the CPU befor making numpy
        predicted_classes = torch.argmax(y_preds, dim=2)  # Shape: [datums, batches]
        true_classes = torch.argmax(y_targs, dim=2) #

        pred_flat = predicted_classes.flatten().cpu().numpy() # Shape: [datums*batches]
        clas_flat = true_classes.flatten().cpu().numpy()
        #print(f"flat pred: {pred_flat.shape}, class: {clas_flat.shape}")

        cm = confusion_matrix(clas_flat, pred_flat)

        # save to log
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(["-------", "Confusion Matrix"])
            for row in cm: writer.writerow(row)

        #logger.info(f"Confusion Matrix:\n{cm}")

        #plt.figure()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix: {self.hyperPeramStr}')
        plt.savefig(f"output/{self.dateTime_str}_validation.jpg")
        plt.show()
        
        return #test_loss, test_acc