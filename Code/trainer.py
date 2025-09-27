###
# Footfal
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Trainer
###

from timeit import default_timer as timer
import numpy as np
from tqdm import tqdm  #progress bar
import matplotlib.pyplot as plt
import csv
import sys
#pyTorch
import torch
from torch import nn
import torch.nn.functional as tFun
import torch.optim as optim
import time
from utils import runStats


#from dataLoader import dataSetWithSubjects
from cwtTransform import cwt

from typing import TYPE_CHECKING
if TYPE_CHECKING: #Fix circular import
    from fileStructure import fileStruct
    from dataLoader import dataLoader
    
import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,model, device, dataPrep:"dataLoader", fileStru:"fileStruct", configs, expNum, 
                 cwtClass:cwt, scaleStr, lossFunction, optimizer, learning_rate, weight_decay, gradiant_noise, epochs):

        logger.info(f"Initializing Trainer: device: {device}, model: {model.__class__.__name__}, loss: {lossFunction}, optimizer: {optimizer}, lr: {learning_rate}, weight_decay: {weight_decay}, gradiant_noise: {gradiant_noise}, epochs: {epochs}")
        
        self.dataPrep = dataPrep
        self.device = device
        self.model = model.to(self.device)
        #self.train_data_loader = self.dataPrep.dataLoader_t 
        #self.val_data_loader   = self.dataPrep.dataLoader_v

        self.expNum = expNum

        self.configs = configs
        #self.outputDir = configs['outputDir']
        #self.modelName = configs['model']['name']


        self.lossFunctionName = lossFunction
        self.optimizerName = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradiant_noise = gradiant_noise
        self.epochs = epochs
        

        #self.batchSize = self.configs['trainer']['batchSize']
        self.classes = self.configs['data']['classes']

        self.regression = self.configs['model']['regression']
        if self.regression:
            self.accStr = f"accuracy (RMS Error)"
        else:
            self.accStr = f"accuracy (%)"

        self.logDir = fileStru.expTrackFiles.expNumDir.expTrackDir_Name
        self.sumaryFile = f"{self.logDir}/{fileStru.expTrackFiles.expNumDir.expTrackSum_fileName}"
        self.trainLog = f"{self.logDir}/trainResults_byBatch.csv"
        self.validLog = f"{self.logDir}/valiResults_byEpoch.csv"

        torch.manual_seed(configs['trainer']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.set_training_config()
        self.cwtClass = cwtClass
        #self.doCWT = False
        waveletStr = ""
        if cwtClass.wavelet_name != "None":
            #self.doCWT = True
            waveletStr = f"{cwtClass.wavelet_name}, "

        #add if it is folded or no, for time domain
        self.hyperPeramStr = f"exp:{self.expNum}, {waveletStr}scale: {scaleStr}," \
                             f"{self.model.__class__.__name__}, " \
                             f"\n"\
                             f"loss:{self.lossFunctionName}, opt:{self.optimizerName}, lr:{self.learning_rate}, wd:{self.weight_decay} " \
                             f"epochs:{self.epochs}"  
        print(f"Hyper Parameters: {self.hyperPeramStr}", flush=True)

        #Writh the headder for the validation acc by epoch file
        with open(self.validLog, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(["Epoch Num", "Loss", self.accStr, "Acc by Class"])

    def set_training_config(self):
        #print(f"Selected Optimizer = {self.optimizerName}", flush=True)
        if self.optimizerName == "SGD":
            #print(self.model.parameters().shape, flush=True)
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)
        elif self.optimizerName == "Adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("Only SGD is supported for now")

        if self.configs['trainer']['LR_sch'] == 'CosineAnnealingWarmRestarts':
            eta_min = float(self.configs['trainer']['eta_min'])
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                                                            T_0=self.configs['trainer']['T_0'], 
                                                                            T_mult=self.configs['trainer']['T-mult'], 
                                                                            eta_min=eta_min)
        elif self.configs['trainer']['LR_sch'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=3, min_lr=eta_min
)

        ## Loss Functions
        #print(f"Selected Loss Function = {self.lossFunctionName}", flush=True )
        if self.lossFunctionName == "MSE": 
            #print(f"Loss function: Mean Squared Error, L2", flush=True)
            self.criterion = nn.MSELoss()
            self.testCrit = nn.MSELoss()
        elif self.lossFunctionName == "MAE": 
            #print(f"Loss function: Mean Absolute Error, L1",flush=True)
            self.criterion = nn.L1Loss() 
            self.testCrit = nn.L1Loss()
        elif self.lossFunctionName == "Huber": # 
            #print(f"Loss function: Huber", flush=True)
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
        
        torch.autograd.set_detect_anomaly(True)

    def add_gradient_noise(self, model, std=1e-3):
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * std  # Small Gaussian noise
                param.grad += noise

    def train(self, batchSize):
        self.batchSize = batchSize
        self.hyperPeramStr = f"{self.hyperPeramStr}, Batch Size: {batchSize}"
        self.model.train()
        lossArr = []
        accArr = []
        valLossArr = []
        valAccArr = []
        train_predsArr =[] # for confusion matrix

        valAccStats = runStats()

        print(f"Model device: {next(self.model.parameters()).device}", flush=True)

        fieldnames = ['epoch', 'lr', 'batch', 'batch correct', self.accStr, 'loss', 'time(s)']
        with open(self.trainLog, 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
            writer.writeheader()

        #for epoch in range(self.epochs):
        for epoch in tqdm(range(self.epochs), desc="Training Progress", unit="epoch", file=sys.stdout):
            batchNumber = 0
            correct_epoch = 0
            train_loss_epoch, train_acc_epoch = 0, 0
            epoch_StartTime = timer()
            epoch_squared_diff = []

            #for data, labels, subjects  in self.train_data_loader: # Batch
            batchStartTime = time.time()
            for data, labelsSpeed, labelsSubject, subjects, runs, sTimes in tqdm(self.dataPrep.dataLoader_t, desc="Epoch Progress", unit="batch", leave=False, file=sys.stdout):
                dataLoadTime = time.time() - batchStartTime
                #logger.info(f"Data Load time: {dataLoadTime}")
                #logger.info(f" data, shape: {data.shape}, type:{type(data)}, {type(data[0][0][0][0].item())}")

                # Not seting the datanormConst is somehow overwriting it?? Makes no sense
                data, self.dataPrep.dataNormConst = self.dataPrep.scale_data(data=data, writeToLog=False, norm=self.dataPrep.dataNormConst, debug=False)
                #print(f"Label Data type: {type(labelsSpeed)}, {labelsSpeed.shape}, {labelsSpeed.dtype}", flush=True)
                if self.regression:
                    labels, self.dataPrep.labNormConst = self.dataPrep.scale_data(data=labelsSpeed, writeToLog=False, norm=self.dataPrep.labNormConst, debug=False)
                else:
                    labels = labelsSubject # we want ([batch size,])
                #print(f"labels shape: {labels.shape}, dtype: {labels.dtype}", flush=True)

                data = data.to(self.device)
                #print(f"Data device: {data.device}", flush=True)
                labels = labels.to(self.device)

                batchNumber +=1
                #for thisRun  in range(0, n):
                batch_StartTime = timer()

                #print(f"Data Shape: {data.shape}", flush=True)
                self.optimizer.zero_grad()

                # Turn the crank 
                out_pred = self.model(data)
                #logger.info(f"out_pred: {out_pred.shape}, labels: {labels.shape}")
                loss = self.criterion(out_pred, labels)
                if self.gradiant_noise != 0: self.add_gradient_noise(self.model, std=self.gradiant_noise)  # Add small noise to gradients
                loss.backward()
                self.optimizer.step()

                #Batch, input ch, height, width
                #print(f"labels shape: {labels.shape}, dtype: {labels.dtype}", flush=True)
                #print(f"Batch Labels: {labels}", flush=True)
                #print(f"output shape: {out_pred.shape}, dtype: {out_pred.dtype}", flush=True)

                ## The Accuracy 
                if self.regression:
                    #print(f"Regression: {out_pred.shape}, {labels.shape}", flush=True)
                    # Calculate RMS accuracy between predictions and labels
                    # Output of unScale is numpy
                    preds_unSc = self.dataPrep.unScale_data(out_pred.squeeze().detach().cpu().numpy(), self.dataPrep.labNormConst)
                    targs_unSc = self.dataPrep.unScale_data(labels.squeeze().detach().cpu().numpy(), self.dataPrep.labNormConst)

                    diff_sq = np.square(preds_unSc - targs_unSc)
                    rms_diff_sq = np.mean(diff_sq)
                    rms_diff = np.sqrt(rms_diff_sq)
                    #print(f"Regression diff: {diff.shape}, rms_diff: {rms_diff}", flush=True)
                    thisAcc = rms_diff
                    # keep track of the squared diffs for the epoch
                    epoch_squared_diff = np.append(epoch_squared_diff, diff_sq)
                    correct_batch = 0
                else:
                    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=len(self.classes)).float()
                    out_pred_argMax = torch.argmax(out_pred, 1) # Convert to argMax
                    labels_argMax = torch.argmax(labels_one_hot,1) #convert to argMax
                    correct_batch = out_pred_argMax.eq(labels_argMax).sum().item() # How many did we get right in this batch
                    correct_epoch += correct_batch
                #print(f"Training Predicted Shape: {out_pred_argMax.shape}")
                #print(f"Training Predicted argMax: {out_pred_argMax}, labels: {labels_argMax}")
                #print(f"Labels: {labels_argMax}")

                train_loss_epoch += loss.item()
                batch_Time = timer() - batch_StartTime

                # Write a file
                if not self.regression:
                    thisAcc = 100*correct_batch/self.batchSize

                with open(self.trainLog, 'a', newline='') as csvFile:
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

                batchStartTime = time.time()

                #End  Batch

            ## Now in Epoch
            valEveryNEpochs = self.configs['trainer']['validEveryNEpochs']
            if self.configs['trainer']['LR_sch'] == "ReduceLROnPlateau": valEveryNEpochs = 1
            if epoch%valEveryNEpochs == 0 and epoch >= self.configs['trainer']['epochValiStart']:
                valLoss, valAcc, classAcc = self.validation(epochNum=epoch) # val acc and loss is printed here
                self.model.train() # Put the model back in train
                valLossArr.append(valLoss)
                valAccArr.append(valAcc)
                if self.epochs - epoch -1 < self.configs['trainer']['nEpochsStats']: # Starting from epoch 0
                    valAccStats.addElement(valAcc)
                    #print(f"add to stats epoch: {epoch}, acc: {valAcc}")
                #print(f" - {epoch} valAccArr: {valAccArr}")

            if self.configs['trainer']['LR_sch'] == 'ReduceLROnPlateau':
                self.scheduler.step(valLoss) # Should be val loss, but use train loss in a pinch
            elif self.configs['trainer']['LR_sch'] != 'None': # or epoch < 10:
                self.scheduler.step() 

            if self.regression:
                train_acc_epoch = np.sqrt(np.mean(epoch_squared_diff))
                accArr.append(train_acc_epoch)
            else:
                train_acc_epoch = 100 * correct_epoch / (batchNumber*self.batchSize )
                accArr.append(train_acc_epoch*100)
            train_loss_epoch = train_loss_epoch/batchNumber
            lossArr.append(train_loss_epoch)
        
            #Timing
            epoch_runTime = timer() - epoch_StartTime
            #print(f"Correct: = {correct_epoch}, nRun: {batchNumber}")
            #logger.debug.. wtf are my loggers not working
            print(f"Training Epoch: {epoch} | LR = {self.optimizer.param_groups[0]['lr']} | " \
                f"Train Loss: {train_loss_epoch:.4f} | {self.accStr}: {train_acc_epoch:.4f} | " \
                #f"Validation Loss: {valLoss:.4f} | {self.accStr}: {valAcc:.4f} | " \
                f"Time: {epoch_runTime:.1f}s")

            # Log the per/epoch results 
            with open(self.trainLog, 'a', newline='') as csvFile:
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
                writer.writerow({'epoch'    : epoch,
                                 'lr'       : self.optimizer.param_groups[0]['lr'],
                                 self.accStr: train_acc_epoch, 
                                 'loss'     : train_loss_epoch,
                                 'time(s)'  : epoch_runTime
                                 })
                
        # End Epochs
        valAccStats.finish()
        print(f"Last {self.configs['trainer']['nEpochsStats']} epochs| min: {valAccStats.min}, max: {valAccStats.max}, mean: {valAccStats.mean}, std: {valAccStats.std}", flush=True)

        self.plotLossAcc(lossArr=lossArr, accArr=accArr)
        self.plotLossAcc(lossArr=valLossArr, accArr=valAccArr, validation=True)
        #print(f"valAccArr: {valAccArr}")

        return train_loss_epoch, train_acc_epoch, valAccStats

    def plotLossAcc(self, lossArr, accArr, validation=False):
        #print(f"Loss shape: {len(lossArr)}")
        nPlots = 2
        fig, axis = plt.subplots(nPlots, 1)
        fig.subplots_adjust(top = 0.90, hspace = .05, left= 0.125, right = 0.99)
        if validation: 
            trainOrVal_str = "V"
            plotPerCount = self.configs['trainer']['validEveryNEpochs']
        else:          
            trainOrVal_str = "T"
            plotPerCount = 1

        #x_values = range(len(accArr))  # Original x-axis indices
        #x_labels = [i * plotPerCount for i in x_values]  # Scale x-axis labels
        axis[0].plot(range(len(lossArr)), lossArr)    
        # Define tick positions based on the multiple (plotPerCount)
        if validation:
            tick_positions = range(0, len(lossArr), plotPerCount)  # Ensure it's within the range of lossArr
            tick_labels = [i * plotPerCount for i in tick_positions] # Define corresponding tick labels (scaled by plotPerCount)

            # Apply to the x-axis
            axis[0].set_xticks(tick_positions)  # Set tick positions
            axis[0].set_xticklabels(tick_labels)  # Set labels to reflect scaling

        axis[0].set_title(f"{trainOrVal_str}: {self.hyperPeramStr}")
        axis[0].set_ylabel(f"{trainOrVal_str} Loss by Epoch")
        axis[0].get_xaxis().set_visible(False)

        #axis[1].plot(x_labels, lossArr)    
        axis[1].plot(range(len(accArr)), accArr)    
        if validation:
            axis[1].set_xticks(tick_positions)  # Set tick positions
            axis[1].set_xticklabels(tick_labels)  # Set labels to reflect scaling
        axis[1].set_ylabel(self.accStr)
        axis[1].get_xaxis().set_visible(True)
        axis[1].set_xlabel("Epoch Number")

        plt.savefig(f"{self.logDir}/loss_acc_{trainOrVal_str}_{self.expNum}.jpg")
        #plt.show()
    
    def validation(self, epochNum=0):
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
            nData = len(self.dataPrep.dataLoader_v)
            #print(f"Test Data len: {nData}")

            for data, labelsSpeed, labelsSubject, subjects, runs, sTimes in tqdm(self.dataPrep.dataLoader_v, desc=f"Validation Progress epoch: {epochNum}", unit="Time Window", file=sys.stdout ):
                # Not seting the datanormConst is somehow overwriting it?? Makes no sense
                data, self.dataPrep.dataNormConst = self.dataPrep.scale_data(data=data, writeToLog=False, norm=self.dataPrep.dataNormConst, debug=False)
                if self.regression:
                    labels, self.dataPrep.labNormConst = self.dataPrep.scale_data(data=labelsSpeed, writeToLog=False, norm=self.dataPrep.labNormConst, debug=False)
                    #logger.info(f"labels shape: {labels.shape}")
                else:
                    labels = tFun.one_hot(labelsSubject, num_classes=len(self.classes))

                data = data.to(self.device)
                labels = labels.to(self.device)

                val_pred = self.model(data)

                #for logging and plotting
                #logger.info(f"val_preds: {type(y_preds)}, {len(y_preds)}, labels: {type(y_targs)}, {len(y_targs)}")
                if self.regression:
                    val_loss = self.criterion(val_pred, labels)
                    # Unscale the data
                    preds_unSc = self.dataPrep.unScale_data(val_pred.item(), self.dataPrep.labNormConst, debug=False)
                    targs_unSc = self.dataPrep.unScale_data(labels.item(), self.dataPrep.labNormConst, debug=False)
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
                print(f"Each Class: {self.accStr}: {classAcc}")
            else: 
                y_preds= torch.stack(y_preds, dim=0) # datums, batch, classes
                y_targs= torch.stack(y_targs, dim=0)
                y_preds = y_preds.view(y_preds.shape[0], -1)  # [datusm*batch, classes]Keeps batch size, removes extra dimension
                y_targs = y_targs.view(y_targs.shape[0], -1)  # Keeps batch size, removes extra dimension
                print(f"pred: {y_preds.shape}, lab: {y_targs.shape} ")

            print(f"Validation Loss: {finalValLoss:.3f} | {self.accStr}: {test_acc:.4}")

        if self.regression:
            if(self.configs['debugs']['writeValData']): self.logRegression(y_preds, y_targs, epochNum=epochNum)
            self.plotRegRes(y_preds, y_targs, valAcc= test_acc, epochNum=epochNum)
        else:
            if(self.configs['debugs']['writeValData']): self.logClassification(y_preds, y_targs, epochNum=epochNum)
            self.plotConfMat(y_preds, y_targs, valAcc=test_acc, epochNum=epochNum)

        with open(self.validLog, 'a', newline='') as csvFile:
            # Write the results after each validation
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow([epochNum, finalValLoss, test_acc, classAcc])

        return finalValLoss, test_acc, classAcc

    #TODO: Put the subject, run, and time info in the log 
    def logRegression(self, y_preds, labels, epochNum=0):
        #logger.info(f"log results: {len(y_preds)}")#, y_preds_targets: {type(y_preds_targets)}")
        with open(f"{self.logDir}/{epochNum}_validationResults_{self.expNum}.csv", 'w', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(['Predictions', 'Labels'])
            for pred, label in zip(y_preds, labels):
                writer.writerow([pred, label])

    def logClassification(self, y_preds, y_targs, epochNum=0):
        y_preds_targets = torch.cat((y_preds, y_targs), dim=1)
        print(f"pred: {y_preds.shape}, targ: {y_targs.shape}, combined: {y_preds_targets.shape}")
        with open(f"{self.logDir}/{epochNum}_validationResults_{self.expNum}.csv", 'w', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(['Predictions', '', '','', 'Labels']) #TODO: fix for n classes
            writer.writerow(self.classes + self.classes)
            for row in y_preds_targets:
                writer.writerow(row.tolist())
    
    def plotRegRes(self, preds, targets, valAcc, epochNum=0):
        #logger.info(f"Type preds: {type(preds[0])}, targets: {type(targets[0])}")
        #logger.info(f"plot results: {len(preds)}, labels: {len(targets)}")
        plt.figure(figsize=(8, 6))
        plt.title(f"V: {self.hyperPeramStr}, \nepoch: {epochNum}, Val Acc: {valAcc:.4f}rms")
        plt.plot(range(len(preds)), preds, label=f"Predictions")    
        plt.plot(range(len(targets)), targets, label=f"targets")    
        plt.legend()
        #plt.plot(targets, preds)    
        plt.xlabel('Validation Test #')
        plt.ylabel('Speed (m/s)')
        plt.savefig(f"{self.logDir}/{epochNum}_validation_{self.expNum}.jpg")
        #plt.show()
        plt.close()

    def plotConfMat(self, y_preds, y_targs, valAcc , epochNum=0):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn  as sns
        # Move back to the CPU befor making numpy
        #predicted_classes = torch.argmax(y_preds, dim=2)  # Shape: [datums, batches]
        #true_classes = torch.argmax(y_targs, dim=2) #
        pred_classes = torch.argmax(y_preds, dim=1)  # Shape: [datums* batches]
        true_classes = torch.argmax(y_targs, dim=1) #

        #pred_flat = predicted_classes.flatten().cpu().numpy() # Shape: [datums*batches]
        #clas_flat = true_classes.flatten().cpu().numpy()
        print(f"flat pred: {pred_classes.shape}, class: {true_classes.shape}")

        cm = confusion_matrix(true_classes, pred_classes)
        logger.info(f"Confusion Matrix:\n{cm}")

        # save to log
        with open(self.sumaryFile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(["-------", "Confusion Matrix", f"Epoch Number: {epochNum}"])
            for row in cm: writer.writerow(row)


        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'CM: {self.hyperPeramStr}, EpochNum: {epochNum}, val acc: {valAcc:.1f}%')
        plt.savefig(f"{self.logDir}/{epochNum}_validation_{self.expNum}.jpg")
        #plt.show()
        plt.close()
        