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
from tqdm import tqdm  #progress bar

import matplotlib.pyplot as plt
import csv

#from dataLoader import dataSetWithSubjects

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,model, device, dataPrep,  configs, logFile, logDir, expNum, waveletName, scaleStr, lossFunction, optimizer, learning_rate, weight_decay, epochs):
        self.device = device

        self.dataPrep = dataPrep
        self.model = model.to(self.device)
        self.train_data_loader = self.dataPrep.dataLoader_t 
        self.val_data_loader   = self.dataPrep.dataLoader_v

        self.expNum = expNum

        self.configs = configs
        #self.outputDir = configs['outputDir']
        #self.modelName = configs['model']['name']


        self.lossFunctionName = lossFunction
        self.optimizerName = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.batchSize = self.configs['data']['batchSize']
        self.classes = self.configs['data']['classes']

        self.regression = self.configs['model']['regression']
        if self.regression:
            self.accStr = f"accuracy (RMS Error)"
        else:
            self.accStr = f"accuracy (%)"

        self.logDir = logDir
        self.logfile = logFile

        torch.manual_seed(configs['trainer']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.set_training_config()
        self.hyperPeramStr = f"exp:{self.expNum}, wavelet: {waveletName}, scale: {scaleStr}\n" \
                             f"loss:{self.lossFunctionName}, opt:{self.optimizerName}, lr:{self.learning_rate}, wd:{self.weight_decay}, epochs:{self.epochs}"  
        print(f"Hyper Parameters: {self.hyperPeramStr}")

    def set_training_config(self):
        #print(f"Selected Optimizer = {self.optimizerName}")
        if self.optimizerName == "SGD":
            #print(self.model.parameters().shape)
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)
        elif self.optimizerName == "Adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("Only SGD is supported for now")

        ## Loss Functions
        #print(f"Selected Loss Function = {self.lossFunctionName}")
        if self.lossFunctionName == "MSE": 
            #print(f"Loss function: Mean Squared Error, L2")
            self.criterion = nn.MSELoss()
            self.testCrit = nn.MSELoss()
        elif self.lossFunctionName == "MAE": 
            #print(f"Loss function: Mean Absolute Error, L1")
            self.criterion = nn.L1Loss() 
            self.testCrit = nn.L1Loss()
        elif self.lossFunctionName == "Huber": # 
            #print(f"Loss function: Huber")
            self.criterion = nn.HuberLoss()
            self.testCrit = nn.MSELoss()
        elif self.lossFunctionName == "Sigmoid":
            self.criterion = nn.BCEWithLogitsLoss()
            self.testCrit = nn.BCEWithLogitsLoss()
        elif self.lossFunctionName == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
            self.testCrit = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Unsupported loss function")

    def train(self ):
        self.model.train()
        lossArr = []
        accArr = []
        train_predsArr =[] # for confusion matrix

        fieldnames = ['epoch', 'batch', 'batch correct', self.accStr, 'loss', 'time(s)']

        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
            writer.writeheader()

        #for epoch in range(self.epochs):
        for epoch in tqdm(range(self.epochs), desc="Training", unit="epoch"):
            batchNumber = 0
            correct_epoch = 0
            train_loss_epoch, train_acc_epoch = 0, 0
            epoch_StartTime = timer()
            epoch_squared_diff = []

            #for data, labels, subjects  in self.train_data_loader: # Batch
            for data, labels, subjects in tqdm(self.train_data_loader, desc="Batch Progress", unit="batch", leave=False):
                data = data.to(self.device)
                labels = labels.to(self.device)

                batchNumber +=1
                #for thisRun  in range(0, n):
                batch_StartTime = timer()

                #print(f"Data Shape: {data.shape}")
                self.optimizer.zero_grad()

                # Turn the crank 
                out_pred = self.model(data)
                #logger.info(f"out_pred: {out_pred.shape}, labels: {labels.shape}")
                loss = self.criterion(out_pred, labels)
                loss.backward()
                self.optimizer.step()

                #Batch, input ch, height, width
                #print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")
                #print(f"Batch Labels: {labels}")
                #print(f"output shape: {out_pred.shape}, dtype: {out_pred.dtype}")

                ## The Accuracy 
                if self.regression:
                    #print(f"Regression: {out_pred.shape}, {labels.shape}")
                    # Calculate RMS accuracy between predictions and labels
                    preds_unSc = self.dataPrep.unScale_data(out_pred.squeeze().detach().cpu().numpy(), self.dataPrep.labNormConst)
                    targs_unSc = self.dataPrep.unScale_data(labels.squeeze().detach().cpu().numpy(), self.dataPrep.labNormConst)
                    #preds_unSc = out_pred.squeeze().detach().cpu().numpy()
                    #targs_unSc = labels.squeeze().detach().cpu().numpy()
                    # Calculate RMS difference between predictions and targets
                    diff_sq = np.square(preds_unSc - targs_unSc)
                    rms_diff_sq = np.mean(diff_sq)
                    rms_diff = np.sqrt(rms_diff_sq)
                    #print(f"Regression diff: {diff.shape}, rms_diff: {rms_diff}")
                    thisAcc = rms_diff
                    # keep track of the squared diffs for the epoch
                    epoch_squared_diff = np.append(epoch_squared_diff, diff_sq)
                    correct_batch = 0
                else:
                    out_pred_argMax = torch.argmax(out_pred, 1) # Convert to argMax
                    labels_argMax = torch.argmax(labels,1) #convert to argMax
                    correct_batch = out_pred_argMax.eq(labels_argMax).sum().item() # How many did we get right in this batch
                    correct_epoch += correct_batch
                #print(f"Training Predicted Shape: {out_pred_argMax.shape}")
                #print(f"Training Predicted argMax: {out_pred_argMax}, labels: {labels_argMax}")
                #print(f"Labels: {labels_argMax}")

                train_loss_epoch += loss.item()
                batch_Time = timer() - batch_StartTime

                '''
                # Echo each batch
                info_str = f"epoch: {epoch}, {batchNumber}"
                pred_str = f"Training Predicted : {out_pred.detach().cpu().numpy()}"
                lab_argMax_str = f"labels: {labels_argMax.detach().cpu().numpy()}"
                pred_argMax_str = f"pred: {out_pred_argMax.detach().cpu().numpy()}"
                percCorr_str = f"correct: {correct_batch}, {100*correct_batch/self.batchSize:.1f}%"
                batchTime_str = f"Batch Time: {batch_Time:.3f}s"
                print(f"{info_str} | {lab_argMax_str} | {pred_argMax_str} | {percCorr_str} | {batchTime_str}")
                '''
                # Write a file
                if not self.regression:
                    thisAcc = 100*correct_batch/self.batchSize

                with open(self.logfile, 'a', newline='') as csvFile:
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
                    writer.writerow({'epoch'            : epoch,
                                     'batch'            : batchNumber,
                                     'batch correct'    : correct_batch, 
                                     self.accStr        : thisAcc,
                                     'loss'             : loss.item(),
                                     'time(s)'          : batch_Time
                                     })

                #print(f"Run correct: {thisTestCorr}, loss: {loss.item()}")
                train_predsArr.append(out_pred) #for confusion matrix
                #End  Batch

            ## Now in Epoch
            batchSize = self.batchSize
            if self.regression:
                train_acc_epoch = np.sqrt(np.mean(epoch_squared_diff))
            else:
                train_acc_epoch = 100 * correct_epoch / (batchNumber*batchSize )
            train_loss_epoch = train_loss_epoch/batchNumber

            lossArr.append(train_loss_epoch)
            accArr.append(train_acc_epoch)
        
            #Timing
            epoch_runTime = timer() - epoch_StartTime
            #print(f"Correct: = {correct_epoch}, nRun: {batchNumber}")
        
            if epoch%1==0:
                print(f"Training Epoch: {epoch+1} | Loss: {train_loss_epoch:.3f} | {self.accStr}: {train_acc_epoch:.2f} | Time: {epoch_runTime:.3f}s")

            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
                writer.writerow({'epoch'    : epoch,
                                 self.accStr: train_acc_epoch, 
                                 'loss'     : train_loss_epoch,
                                 'time(s)'  : epoch_runTime
                                 })
        # End Epoch

        self.plotTrainingLoss(lossArr=lossArr, accArr=accArr )

        return train_loss_epoch, train_acc_epoch

    def plotTrainingLossRegresh(self, lossArr ):
        plt.figure(figsize=(10,5))
        plt.title(f"{self.hyperPeramStr}")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.ylim([0,1])
        plt.plot(lossArr)    
        plt.savefig(f"{self.logDir}/trainingLoss_{self.expNum}.jpg")

    def plotTrainingLoss(self, lossArr, accArr ):
        #print(f"Loss shape: {len(lossArr)}")
        nPlots = 2
        fig, axis = plt.subplots(nPlots, 1)
        fig.subplots_adjust(top = 0.90, hspace = .05, left= 0.125, right = 0.99)
        axis[0].plot(range(len(lossArr)), lossArr)    
        axis[0].set_title(f"{self.hyperPeramStr}")
        axis[0].set_ylabel("Training Loss")
        axis[0].get_xaxis().set_visible(False)

        axis[1].plot(range(len(accArr)), accArr)    
        axis[1].set_ylabel(self.accStr)
        axis[1].get_xaxis().set_visible(True)
        axis[1].set_xlabel("Epoch Number")

        plt.savefig(f"{self.logDir}/trainingLoss_{self.expNum}.jpg")
        #plt.show()
    
    def validation(self):
        self.model.eval()
        test_loss, test_acc, total = 0, 0, 0
        correct,total = 0,0
        y_preds = [] # for confusion matrix
        y_targs = []
        y_sqDif = []
        classAcc = [0] * len(self.classes)
        classNum = [0] * len(self.classes)

        with torch.no_grad():
        #with torch.inference_mode():
            nData = len(self.val_data_loader)
            print(f"Test Data len: {nData}")

            for data, labels, subjects  in self.val_data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                val_pred = self.model(data)

                #for logging and plotting
                #logger.info(f"val_preds: {type(y_preds)}, {len(y_preds)}, labels: {type(y_targs)}, {len(y_targs)}")
                if self.regression:
                    val_loss = self.criterion(val_pred, labels)
                    # Unscale the data
                    preds_unSc = self.dataPrep.unScale_data(val_pred.item(), self.dataPrep.labNormConst)
                    targs_unSc = self.dataPrep.unScale_data(labels.item(), self.dataPrep.labNormConst)
                    #logger.info(f"val_pred: {type(val_pred)}, {val_pred.shape}, labels: {type(labels)}, {labels.shape}")
                    #logger.info(f"val_pred: {val_pred}, labels: {labels}")
                    diff_sq = np.square(preds_unSc - targs_unSc)

                    y_sqDif.append(diff_sq)
                    for i in range(len(self.classes)):
                        if subjects == i:
                            classAcc[i] += diff_sq
                            classNum[i] += 1
                    y_preds.append(preds_unSc)
                    y_targs.append(targs_unSc)
                else:
                    labels_argMax = torch.argmax(labels,1) #convert to argMax
                    labels_argMax = labels_argMax.to(self.device)
                    val_loss = self.criterion(val_pred, labels_argMax)

                    val_pred_argMax = torch.argmax(val_pred, 1) # Convert to argMax
                    correct += val_pred_argMax.eq(labels_argMax).sum().item()
                    #print(f"This pred: {val_pred}, {val_pred_argMax}, lab: {labels}, {labels_argMax}")

                    ## For confusion and logging
                    # We don't want to argmax yet cuz we want to log hotmax... er, do we?
                    y_preds.append(val_pred.detach().cpu()) 
                    y_targs.append(labels.detach().cpu()) 
                    #print(f"This pred: {type(y_preds)}, lab: {type(y_targs)} ")
                    #print(f"This pred: {len(y_preds)}, lab: {len(y_targs)} ")

                test_loss += val_loss.item()
                if not self.regression:
                    test_acc = 100 * correct / (nData)

                # This gets written to a csv
                #print(f"This pred: {val_pred.detach().cpu().numpy()}, lab: {labels_argMax.detach().cpu().numpy()}")
                #print(f"Correct running: {correct}, loss: {test_loss/(thisRun+1):.3f}, accu: {test_acc:.1f}%")

                # End training for loop
            finalValLoss = test_loss/nData
            #print(f"Final Val Predicted: {y_preds}")
            #print(f"Labels: {y_targs}")
            #print(f"y_logigs: {y_argMax}")
            if self.regression:
                test_acc = np.sqrt(np.mean(y_sqDif)) # Overall RMS error
                for i in range(len(self.classes)):
                    if classAcc[i] > 0:  # Only calculate if we have samples for this class
                        classAcc[i] = np.sqrt(classAcc[i]/classNum[i])# Per-class RMS error
                print(f"Class Acc {self.accStr}: {classAcc}")
            else: 
                y_preds= torch.stack(y_preds, dim=0) # datums, batch, classes
                y_targs= torch.stack(y_targs, dim=0)
                y_preds = y_preds.view(y_preds.shape[0], -1)  # [datusm*batch, classes]Keeps batch size, removes extra dimension
                y_targs = y_targs.view(y_targs.shape[0], -1)  # Keeps batch size, removes extra dimension
                print(f"Final pred: {y_preds.shape}, lab: {y_targs.shape} ")

            print(f"Validation Loss: {finalValLoss:.3f} | {self.accStr}: {test_acc:.2f}")

            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile, dialect='unix')
                writer.writerow(["------- Validation --------"])
                writer.writerow(["Loss", self.accStr])
                writer.writerow([finalValLoss, test_acc])
                if self.regression:
                    writer.writerow(["Class Acc: "])
                    for i in range(len(self.classes)):
                        writer.writerow([self.classes[i], classAcc[i]])


        if self.regression:
            if(self.configs['debugs']['writeValData']): self.logRegression(y_preds, y_targs)
            self.plotRegRes(y_preds, y_targs)
        else:
            if(self.configs['debugs']['writeValData']): self.logClassification(y_preds, y_targs)
            self.plotConfMat(y_preds, y_targs)

        return finalValLoss, test_acc, classAcc

    #TODO: Put the subject, run, and time info in the log 
    def logRegression(self, y_preds, labels):
        #logger.info(f"log results: {len(y_preds)}")#, y_preds_targets: {type(y_preds_targets)}")
        with open(f"{self.logDir}/valiResults.csv", 'w', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(['Predictions', 'Labels'])
            for pred, label in zip(y_preds, labels):
                writer.writerow([pred, label])

    def logClassification(self, y_preds, y_targs):
        y_preds_targets = torch.cat((y_preds, y_targs), dim=1)
        print(f"pred: {y_preds.shape}, targ: {y_targs.shape}, combined: {y_preds_targets.shape}")
        with open(f"{self.logDir}/validationResults_{self.expNum}.csv", 'w', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(['Predictions', '', '', 'Labels'])
            writer.writerow(self.classes + self.classes)
            for row in y_preds_targets:
                writer.writerow(row.tolist())
    
    def plotRegRes(self, preds, targets ):
        #logger.info(f"Type preds: {type(preds[0])}, targets: {type(targets[0])}")
        #logger.info(f"plot results: {len(preds)}, labels: {len(targets)}")
        plt.figure(figsize=(8, 6))
        plt.title(f"Regresion Validation Results: {self.hyperPeramStr}")
        plt.plot(range(len(preds)), preds, label=f"Predictions")    
        plt.plot(range(len(targets)), targets, label=f"targets")    
        plt.legend()
        #plt.plot(targets, preds)    
        plt.xlabel('Validation Test #')
        plt.ylabel('Speed (m/s)')
        plt.savefig(f"{self.logDir}/validation_{self.expNum}.jpg")
        #plt.show()

    def plotConfMat(self, y_preds, y_targs ):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn  as sns
        # Move back to the CPU befor making numpy
        #predicted_classes = torch.argmax(y_preds, dim=2)  # Shape: [datums, batches]
        #true_classes = torch.argmax(y_targs, dim=2) #
        predicted_classes = torch.argmax(y_preds, dim=1)  # Shape: [datums* batches]
        true_classes = torch.argmax(y_targs, dim=1) #

        #pred_flat = predicted_classes.flatten().cpu().numpy() # Shape: [datums*batches]
        #clas_flat = true_classes.flatten().cpu().numpy()
        print(f"flat pred: {predicted_classes.shape}, class: {true_classes.shape}")

        cm = confusion_matrix(predicted_classes, true_classes)
        logger.info(f"Confusion Matrix:\n{cm}")

        # save to log
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(["-------", "Confusion Matrix"])
            for row in cm: writer.writerow(row)


        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix: {self.hyperPeramStr}')
        plt.savefig(f"{self.logDir}/validation_{self.expNum}.jpg")
        #plt.show()
        