import logging
import numpy as np
import torch
import pdb
from sklearn.metrics import confusion_matrix

def get_log(file_name):
    logger = logging.getLogger('*')  
    logger.setLevel(logging.INFO)  

    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)  

    fh = logging.FileHandler(file_name, mode='a')  
    fh.setLevel(logging.INFO)  

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    logger.addHandler(ch)
    return logger

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

def makeArrayOneHot(arrayWithClassInts, cardinality, axisInResultToUseForCategories):
    # arrayWithClassInts: np array of shape [batchSize, r, c, z], with sampled ints expressing classes for each of the samples.
    oneHotsArray = np.zeros( [cardinality] + list(arrayWithClassInts.shape), dtype=np.float32 )
    oneHotsArray = np.reshape(oneHotsArray, newshape=(cardinality, -1)) #Flatten all dimensions except the first.
    arrayWithClassInts = np.reshape(arrayWithClassInts, newshape=-1) # Flatten
    
    oneHotsArray[arrayWithClassInts, range(oneHotsArray.shape[1])] = 1
    oneHotsArray = np.reshape(oneHotsArray, newshape=[cardinality] + list(arrayWithClassInts.shape)) # CAREFUL! cardinality first!
    oneHotsArray = np.swapaxes(oneHotsArray, 0, axisInResultToUseForCategories) # in my implementation, axisInResultToUseForCategories == 1 usually.
    
    return oneHotsArray

def onehot_to_label(one_hot):
    # result means the digit label 
    # result = torch.topk(one_hot, 1)[1].squeeze(1)
    result = torch.argmax(one_hot,dim=1)
    return result

def get_confusion_matrix_values(y_true, y_pred,labels):
    cm = confusion_matrix(y_true, y_pred,labels=labels)
    return(cm, cm[0][0], cm[0][1], cm[1][0], cm[1][1])