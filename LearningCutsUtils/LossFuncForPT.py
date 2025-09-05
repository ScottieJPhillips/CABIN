import torch
import numpy as np
from LearningCutsUtils.LossFunctions import lossvars, loss_fn

    
def pt_loss_fn(y_pred, y_true, features, net,
                  alpha=1., beta=1., gamma=0.001, delta=0., epsilon=0.001,
                  debug=False):

    sumptlosses=None    
    for i in range(len(net.pt)):
        pt=net.pt[i][0]
        ptnet = net.nets[i]
        l=loss_fn(y_pred[i], y_true[i], features, 
                  ptnet, 0.8,
                  alpha, beta, gamma, delta, debug)
        if sumptlosses==None:
            sumptlosses=l
        else:
            sumptlosses = sumptlosses + l

    loss=sumptlosses

    sortedpt=sorted(net.pt)

    if len(sortedpt)>=3:
        featureloss = None
        for i in range(1,len(sortedpt)-1):
            cuts_i   = net.nets[i  ].get_cuts()
            cuts_im1 = net.nets[i-1].get_cuts()
            cuts_ip1 = net.nets[i+1].get_cuts()

            fl = None

            cutrange           =  cuts_ip1-cuts_im1
            mean               = (cuts_ip1+cuts_im1)/2.
            distance_from_mean = (cuts_i  -mean)
            
            exponent=2.  
            
            fl=(distance_from_mean**exponent)/((cutrange**exponent)+0.1)
            # -----------------------------------------------------
          
            if featureloss == None:
                featureloss = fl
            else:
                featureloss = featureloss + fl

        sumfeaturelosses = torch.sum(featureloss)/(len(sortedpt)-2)/features
        loss.monotloss = epsilon*sumfeaturelosses

    return loss