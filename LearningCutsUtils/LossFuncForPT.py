import torch
import numpy as np

class lossvars():

    def __init__(self):
        self.efficloss = 0
        self.backgloss = 0
        self.cutszloss = 0
        self.monotloss = 0
        self.BCEloss = 0
        self.signaleffic = 0
        self.backgreffic = 0
    
    def totalloss(self):
        return self.efficloss + self.backgloss + self.cutszloss + self.monotloss + self.BCEloss

    def __add__(self,other):
        third=lossvars()
        third.efficloss = self.efficloss + other.efficloss
        third.backgloss = self.backgloss + other.backgloss
        third.cutszloss = self.cutszloss + other.cutszloss
        third.monotloss = self.monotloss + other.monotloss
        third.BCEloss   = self.BCEloss   + other.BCEloss
        
        if type(self.signaleffic) is list:
            third.signaleffic = self.signaleffic
            third.signaleffic.append(other.signaleffic)
        else:
            third.signaleffic = []
            third.signaleffic.append(self.signaleffic)
            third.signaleffic.append(other.signaleffic)
        if type(self.backgreffic) is list:
            third.backgreffic = self.backgreffic
            third.backgreffic.append(other.backgreffic)
        else:
            third.backgreffic = []
            third.backgreffic.append(self.backgreffic)
            third.backgreffic.append(other.backgreffic)
        return third


def loss_fn (y_pred, y_true, features, 
             net, pt,
             alpha=1., beta=1., gamma=0.001, delta=0.,
             debug=False):

    loss = lossvars()
    
    signal_results = y_pred.detach() * y_true
    loss.signaleffic = torch.sum(signal_results)/np.sum(y_true)

    background_results = y_pred.detach() *(1- y_true)
    loss.backgreffic = torch.sum(background_results)/(np.sum(1.-y_true))

    loss.efficloss = alpha*torch.square(torch.tensor(0.8)-loss.signaleffic)

    loss.backgloss = beta*loss.backgreffic

    cuts=net.get_cuts()
    loss.cutszloss = gamma*torch.sum(torch.square(cuts))/features

    # loss.BCEloss = delta*torch.nn.BCELoss()(y_pred.detach(), y_true)    
    
    if debug:
        print(f"Inspecting efficiency loss: alpha={alpha}, target={0.8:4.3f}, subnet_effic={loss.signaleffic:5.4f}, subnet_backg={loss.backgreffic:5.4f}, efficloss={loss.efficloss:4.3e}, backgloss={loss.backgloss:4.3e}")
    
    return loss
    
def pt_loss_fn(y_pred, y_true, features, net,
                  alpha=1., beta=1., gamma=0.001, delta=0., epsilon=0.001,
                  debug=False):

    sumptlosses=None    
    for i in range(len(net.pt)):
        pt=net.pt[i][0]
        ptnet = net.nets[i]
        l=loss_fn(y_pred[i], y_true[i], features, 
                  ptnet, pt,
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