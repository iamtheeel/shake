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
#import numpy as np

import matplotlib.pyplot as plt
import csv

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,model, device, dataPrep,  configs, logFile, dateTime_str):
    #def __init__(self,model, device, train_data_loader, val_data_loader,  configs, logFile, dateTime_str):
        self.device = device

        self.dataPrep = dataPrep
        self.model = model.to(self.device)
        self.train_data_loader = self.dataPrep.dataLoader_t 
        self.val_data_loader   = self.dataPrep.dataLoader_v

        self.configs = configs
        self.outputDir = configs['outputDir']

        modelName = configs['model']['name']
        self.regression = configs['model']['regression']

        if self.regression: self.critName = configs['trainer']['criterion_regresh']
        else:               self.critName = configs['trainer']['criterion_class']
        self.optimizer = configs['trainer']['optimizer']
        self.epochs = configs['trainer']['epochs']

        self.batchSize = configs['data']['batchSize']
        self.classes = self.configs['data']['classes']


        self.hyperPeramStr = f"{modelName}, batch size:{self.batchSize}, epochs:{self.epochs}"

        self.dateTime_str = dateTime_str
        self.logfile = logFile

        torch.manual_seed(configs['trainer']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        print(f"Selected Loss Function = {self.critName}")
        if self.critName == "MSE": 
            print(f"Loss function: Mean Squared Error, L2")
            self.criterion = nn.MSELoss()
            self.testCrit = nn.MSELoss()
        elif self.critName == "MAE": 
            print(f"Loss function: Mean Absolute Error, L1")
            self.criterion = nn.L1Loss() 
            self.testCrit = nn.L1Loss()
        elif self.critName == "Huber": # 
            print(f"Loss function: Huber")
            self.criterion = nn.HuberLoss()
            self.testCrit = nn.MSELoss()
        elif self.critName == "Sigmoid":
            self.criterion = nn.BCEWithLogitsLoss()
            self.testCrit = nn.BCEWithLogitsLoss()
        elif self.critName == "CrossEntropyLoss":
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
                labels = labels.to(self.device)

                batchNumber +=1
                #for thisRun  in range(0, n):
                batch_StartTime = timer()

                #print(f"Data Shape: {data.shape}")
                self.optimizer.zero_grad()

                # Turn the crank 
                out_pred = self.model(data)
                loss = self.criterion(out_pred, labels)
                loss.backward()
                self.optimizer.step()

                #Batch, input ch, height, width
                #print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")
                #print(f"Batch Labels: {labels}")
                #print(f"output shape: {out_pred.shape}, dtype: {out_pred.dtype}")

                ## The Accuracy 
                if self.regression:
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
                if self.regression:
                    print(f"Epoch: {epoch} | Train Loss: {train_loss_epoch:.3f} | Epoch Time: {epoch_runTime:.3f}s")
                else:
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
        if self.regression:
            self.plotTrainingLossRegresh(lossArr=lossArr)
        else:
            self.plotTrainingLoss(lossArr=lossArr, accArr=accArr)

        return train_loss_epoch, train_acc_epoch

    def plotTrainingLossRegresh(self, lossArr):
        plt.figure(figsize=(10,5))
        plt.title(f"{self.hyperPeramStr}")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.ylim([0,1])
        plt.plot(lossArr)    
        plt.savefig(f"output/{self.dateTime_str}/{self.dateTime_str}_trainingLoss.jpg")

    def plotTrainingLoss(self, lossArr, accArr):
        #print(f"Loss shape: {len(lossArr)}")
        nPlots = 2
        #if self.regression: nPlots = 1
        fig, axis = plt.subplots(nPlots, 1)
        fig.subplots_adjust(top = 0.92, hspace = .05, left= 0.125, right = 0.99)
        axis[0].plot(range(len(lossArr)), lossArr)    
        axis[0].set_title(f"{self.hyperPeramStr}")
        axis[0].set_ylabel("Training Loss")
        axis[0].get_xaxis().set_visible(False)

        if not self.regression:
            axis[1].plot(range(len(accArr)), accArr)    
            axis[1].set_ylabel("Accuracy (%)")
            axis[1].get_xaxis().set_visible(True)
            axis[1].set_xlabel("Epoch")

        plt.savefig(f"output/{self.dateTime_str}/{self.dateTime_str}_trainingLoss.jpg")
        #plt.show()
    
    def validation(self):
        self.model.eval()
        test_loss, test_acc, total = 0, 0, 0
        correct,total = 0,0
        y_preds =[] # for confusion matrix
        y_targs = []

        with torch.no_grad():
        #with torch.inference_mode():
            nData = len(self.val_data_loader)
            print(f"Test Data len: {nData}")

            for data, labels  in self.val_data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                val_pred = self.model(data)

                #for logging and plotting
                #logger.info(f"val_preds: {type(y_preds)}, {len(y_preds)}, labels: {type(y_targs)}, {len(y_targs)}")
                if self.regression:
                    val_loss = self.criterion(val_pred, labels)
                    #logger.info(f"val_pred: {type(val_pred)}, {val_pred.shape}, labels: {type(labels)}, {labels.shape}")
                    #logger.info(f"val_pred: {val_pred}, labels: {labels}")
                    y_preds.append(val_pred.item())
                    y_targs.append(labels.item())
                else:
                    labels_argMax = torch.argmax(labels,1) #convert to argMax
                    labels_argMax = labels_argMax.to(self.device)
                    val_loss = self.criterion(val_pred, labels_argMax)

                    val_pred_argMax = torch.argmax(val_pred, 1) # Convert to argMax
                    #labels_argMax = torch.argmax(labels,1) #convert to argMax
                    correct += val_pred_argMax.eq(labels_argMax).sum().item()
                    #print(f"This pred: {val_pred}, lab: {labels}, {labels_argMax}")
                    #y_preds.append(val_pred) #for confusion matrix
                    #y_targs.append(labels) #for confusion matrix 
                    # Convert to torch tensor
                    y_preds.append(val_pred) 
                    y_targs.append(labels) 
                    y_preds = torch.stack(y_preds) # datums, batch, classes
                    y_targs = torch.stack(y_targs)

                test_loss += val_loss.item()
                test_acc = 100 * correct / (nData+1)


                # This gets written to a csv
                #print(f"This pred: {val_pred.detach().cpu().numpy()}, lab: {labels_argMax.detach().cpu().numpy()}")
                #print(f"Correct running: {correct}, loss: {test_loss/(thisRun+1):.3f}, accu: {test_acc:.1f}%")


            finalValLoss = test_loss/nData

            #print(f"Final Val Predicted: {y_preds}")
            #print(f"Labels: {y_targs}")
            #print(f"y_logigs: {y_argMax}")

            print(f"Validation Loss: {finalValLoss:.3f} | Test Acc: {test_acc:.2f}%")
            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile, dialect='unix')
                writer.writerow(["-------", "Validation"])
                writer.writerow(["Loss", "Accuracy"])
                writer.writerow([finalValLoss, test_acc])


        if self.regression:
            # descale the data, note, converts to numpy
            y_preds = self.dataPrep.unScale_data(y_preds, self.dataPrep.labNormConst)
            y_targs = self.dataPrep.unScale_data(y_targs, self.dataPrep.labNormConst)
            if(self.configs['debugs']['writeValData']): self.logRegression(y_preds, y_targs)
            self.plotRegRes(y_preds, y_targs)
        else:
            if(self.configs['debugs']['writeValData']): self.logClassification(y_preds, y_targs)
            self.plotConfMat(y_preds, y_targs)

        return finalValLoss, test_acc
    
    def logRegression(self, y_preds, labels):
        logger.info(f"log results: {len(y_preds)}, labels: {len(labels)}")

    def logClassification(self, y_preds, y_targs):
            nClasses = len(self.classes)
            y_preds_flt = y_preds.view(-1,nClasses)
            y_targs_flt = y_targs.view(-1,nClasses)
            y_preds_targets = torch.cat((y_preds_flt, y_targs_flt), dim=1)
            print(f"pred: {y_preds_flt.shape}, targ: {y_targs_flt.shape}, combined: {y_preds_targets.shape}")
            with open(f"{self.configs['outputDir']}/{self.dateTime_str}/{self.dateTime_str}_valiResults.csv", 'w', newline='') as csvFile:
                writer = csv.writer(csvFile, dialect='unix')
                writer.writerow(['Predictions', '', '', 'Labels'])
                writer.writerow(self.classes + self.classes)
                for row in y_preds_targets:
                    writer.writerow(row.tolist())
    
    def plotRegRes(self, preds, targets):
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
        plt.savefig(f"output/{self.dateTime_str}/{self.dateTime_str}_validation.jpg")
        plt.show()

    def plotConfMat(self, y_preds, y_targs):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn  as sns
        # Move back to the CPU befor making numpy
        predicted_classes = torch.argmax(y_preds, dim=2)  # Shape: [datums, batches]
        true_classes = torch.argmax(y_targs, dim=2) #

        pred_flat = predicted_classes.flatten().cpu().numpy() # Shape: [datums*batches]
        clas_flat = true_classes.flatten().cpu().numpy()
        print(f"flat pred: {pred_flat.shape}, class: {clas_flat.shape}")

        cm = confusion_matrix(clas_flat, pred_flat)

        # save to log
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(["-------", "Confusion Matrix"])
            for row in cm: writer.writerow(row)

        logger.info(f"Confusion Matrix:\n{cm}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix: {self.hyperPeramStr}')
        plt.savefig(f"output/{self.dateTime_str}/{self.dateTime_str}_validation.jpg")
        plt.show()
        