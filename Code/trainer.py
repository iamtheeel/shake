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

import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,model, train_data, train_labels, val_data, val_labels, configs):
        self.model = model
        self.train_data = torch.from_numpy(train_data).float()
        self.train_labels = train_labels
        #self.train_labels = torch.from_numpy(train_labels).long()
        self.val_data = torch.from_numpy(val_data).float()
        self.val_labels = val_labels
        self.configs = configs

        self.optimizer = configs['trainer']['optimizer']
        self.criterion = configs['trainer']['criterion']
        self.learning_rate = configs['trainer']['learning_rate']
        self.epochs = configs['trainer']['epochs']

        self.set_training_config()

        logger.info(f"Train data: {type(train_data)}, {train_data.dtype}, {train_data.shape}")
        logger.info(f"Train labels: {type(train_labels)}, {train_labels.dtype}, {train_labels.shape}")


    def set_training_config(self):
        print(f"Selected Optimizer = {self.optimizer}")
        if self.optimizer == "SGD":
            print(f"Setting opt = SGD")
            #print(self.model.parameters().shape)
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
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
        

        for epoch in range(self.epochs):
            #for thisRun  in range(0, n):
            train_loss, train_acc = 0, 0
            correct, total = 0,0
            #runStartTime = timer()

            #image = image.to(self.device)
            #label = label.to(self.device)
    
            out_pred = self.model(self.train_data)
            #out_pred = self.model(self.train_data[thisRun])

            #print(f"Data: {self.train_data.shape}")
            #print(f"Data: {self.train_data}")
            #print(f"Output shape: {out_pred.shape}, dtype: {out_pred.dtype}")
            #print(f"Labels shape: {self.train_labels.shape}, dtype: {self.train_labels.dtype}")
            loss = self.criterion(out_pred, self.train_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ## The Accuracy 
            #print(f"Training Predicted : {out_pred.shape}")
            #print(f"Training Predicted : {out_pred}")
            out_pred_logits = torch.argmax(out_pred, 1) # Convert to logits
            labels_logits = torch.argmax(self.train_labels,1) #convert to logits
            #print(f"Training Predicted Shape: {out_pred_logits.shape}")
            #print(f"Training Predicted logits: {out_pred_logits}")
            #print(f"Labels: {labels_logits}")
            total += self.train_labels.size(0)
            correct += out_pred_logits.eq(labels_logits).sum().item()
            train_loss += loss.item()
            train_acc = 100 * correct / total
        
            #Timing
            #runEndTime = timer()
            #runTime = runEndTime - runStartTime

            lossArr.append(train_loss)
            accArr.append(train_acc)
    
            if epoch%1==0:
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}")
                #print(f"Epoch: {epoch} | Train Loss: {thisTrainLoss:.3f} | Train Acc: {train_acc:.2f} | Elapsed Time: {runTime}")

        print(f"Final Training Predicted logits: {out_pred_logits}")
        print(f"Labels: {labels_logits}")

        print(f"Loss shape: {len(lossArr)}")
        thisFig, axis = plt.subplots(2, 1)
        axis[0].plot(range((epoch+1)), lossArr)    
        axis[0].set_title("Training loss v Batch")

        axis[1].plot(range((epoch+1)), accArr)    
        axis[1].set_title("Accuracy v Batch")

        #plt.savefig("output/trainingLoss.png")
        #plt.show()
        return train_loss, accArr# Return the final training loss

    def validation(self):
        self.model.eval()
        test_loss, test_acc, total = 0, 0, 0
        correct,total = 0,0
        y_preds =[] # for confusion matrix
        y_logits =[] # for confusion matrix
        y_targs = []

        with torch.inference_mode():
            print(f"Test Data len: {self.val_data.shape}")
            print(f"Test Lables: {self.val_labels.shape}")

            numDatam = self.val_data.shape[0] 
            for thisRun  in range(0, numDatam):
                thisData = self.val_data[thisRun].unsqueeze(0) # Make into batch of 1
                thisLabel = self.val_labels[thisRun].unsqueeze(0) 
                #print(f"This Data len: {thisData.shape}")
                #print(f"This Lable: {self.val_labels[thisRun].shape}")

                val_pred = self.model(thisData)
                val_loss = self.criterion(val_pred, thisLabel)

                #print(f"This pre: {val_pred}")
                val_pred_logits = torch.argmax(val_pred, 1) # Convert to logits
                #print(f"This lab: {thisLabel}")
                labels_logits = torch.argmax(thisLabel,1) #convert to logits

                print(f"Val Predicted logits: {val_pred_logits[0]}")
                print(f"Labels: {labels_logits[0]}")

                correct += val_pred_logits.eq(labels_logits).sum().item()
                test_loss += val_loss.item()
                test_acc = 100 * correct / (thisRun+1)

                y_preds.append(val_pred) #for confusion matrix
                y_logits.append(val_pred_logits )
                y_targs.append(thisLabel) #for confusion matrix 

                print(f"Correct: {correct}, loss: {test_loss}, accu: {test_acc}%")


            finalTestLoss = test_loss/numDatam

            print(f"Final Val Predicted logits: {y_preds}")
            print(f"Labels: {y_targs}")
            print(f"y_logigs: {y_logits}")

            print(f"Test Loss: {finalTestLoss:.3f} | Test Acc: {test_acc:.2f}%")

        
        #from torchmetrics import ConfusionMatrix
        #from mlxtend.plotting import plot_confusion_matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn  as sns

        import sys
        print(sys.path)
        print(type(plt))  # Should output <class 'module'>

        y_preds = torch.stack(y_preds).squeeze(1)
        y_targs = torch.stack(y_targs).squeeze(1)

        print(f"y_preds: {y_preds.shape}")
        print(f"y_targs: {y_targs.shape}")

        predicted_classes = torch.argmax(y_preds, dim=1)  # Shape: [13]
        true_classes = torch.argmax(y_targs, dim=1)  # Shape: [13]

        ## Now to Numpy, this is silly
        true_classes = true_classes.numpy()  
        predicted_classes = predicted_classes.numpy()

        print(f"y_preds: {type(predicted_classes)}")
        print(f"y_targs: {type(true_classes)}")

        print(f"y_preds.shape: {y_preds.shape}")  # Should be [13, num_classes]
        print(f"y_targs.shape: {y_targs.shape}")  # Should be [13, num_classes]
        print(f"predicted_classes.shape: {predicted_classes.shape}")  # Should be [13]
        print(f"true_classes.shape: {true_classes.shape}")  # Should be [13]

        cm = confusion_matrix(true_classes, predicted_classes)
        print(f"Confusion Matrix: {cm}")

        #plt.figure()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        '''
        classes = ['001', '002', '003']
        y_pred_tensor = torch.cat(y_preds)
        y_targ_tensor = torch.cat(y_targs)
        print(f"y_preds: {y_pred_tensor.shape}")
        print(f"y_targs: {y_targ_tensor.shape}")

        confMat = ConfusionMatrix(num_classes=len(classes), task='multiclass')
        confMat_values = confMat(preds=y_pred_tensor, target=y_targ_tensor)
        print(f"confustion Matix: {confMat_values}")

        plot_confusion_matrix(conf_mat=confMat_values.numpy(), class_names=classes)
        #plt.title("Confusion Matrix")
        #plt.savefig("../output/confMatrix.png")
        plt.show()
        '''
        
        return test_loss, test_acc