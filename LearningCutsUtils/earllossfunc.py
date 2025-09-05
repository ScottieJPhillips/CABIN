import torch


class lossvars():

    def __init__(self):
        self.efficloss = 0
        self.backgloss = 0
        self.cutszloss = 0
        self.ptloss = 0
        self.muloss = 0
        self.efficfeatloss = 0
        self.signaleffic = 0
        self.backgreffic = 0
    
    def totalloss(self):
        return self.efficloss + self.backgloss + self.cutszloss + self.ptloss + self.muloss + self.efficfeatloss

    def __add__(self,other):
        third=lossvars()
        third.efficloss = self.efficloss + other.efficloss
        third.backgloss = self.backgloss + other.backgloss
        third.cutszloss = self.cutszloss + other.cutszloss
        third.ptloss    = self.ptloss    + other.ptloss
        third.muloss    = self.muloss    + other.muloss
        third.efficfeatloss = self.efficfeatloss + other.efficfeatloss

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
             net, target_signal_efficiency,
             alpha=1., beta=1., gamma=0.001, delta = 0.,
             debug=False):

    loss = lossvars()
    
    # signal efficiency: (selected events that are true signal) / (number of true signal)
    signal_results = y_pred * y_true
    loss.signaleffic = torch.sum(signal_results)/torch.sum(y_true)

    # background efficiency: (selected events that are true background) / (number of true background)
    background_results = y_pred * (1.-y_true)
    loss.backgreffic = torch.sum(background_results)/(torch.sum(1.-y_true))

    cuts=net.get_cuts()
    
    # * force signal efficiency to converge to a target value
    # * force background efficiency to small values at target efficiency value.
    # * also prefer to have the cuts be close to zero, so they're not off at some crazy 
    #   value even if we prefer for the cut to not have much impact on the efficiency 
    #   or rejection.
    #
    # should modify the efficiency target requirement here, to make this more 
    # like consistency with e.g. a gaussian distribution rather than just a penalty 
    # calculated from r^2 distance.
    #
    # for both we should prefer to do something like "sum(square())" or something.
    loss.efficloss = alpha*torch.square(target_signal_efficiency-loss.signaleffic)
    loss.backgloss = beta*loss.backgreffic
    loss.cutszloss = gamma*torch.sum(torch.square(cuts))/features

    if debug:
        print(f"Inspecting efficiency loss: alpha={alpha}, target={target_signal_efficiency:4.3f}, subnet_effic={loss.signaleffic:5.4f}, subnet_backg={loss.backgreffic:5.4f}, efficloss={loss.efficloss:4.3e}, backgloss={loss.backgloss:4.3e}")
    
    # sanity check in case we ever need it, should work
    #loss=bce_loss_fn(outputs_to_labels(y_pred,features),y_true)
    
    return loss


    

def full_loss_fn(y_pred, y_true, features, net,
                  alpha=1., beta=1., gamma=0.001, delta=0., epsilon=0.001,
                  debug=False):
    sumptlosses=None  
    # print(len(net.eta))
    for i in range(len(net.eta)):
        for j in range(len(net.pt)):
                subnet = net.nets[i][j]
                l=loss_fn(y_pred[i][j], y_true[i][j], features, 
                          subnet, 0.8,
                          alpha, beta, gamma, delta, debug)
                if sumptlosses==None:
                    sumptlosses=l
                else:
                    sumptlosses = sumptlosses + l

    loss=sumptlosses

    

    if len(net.pt)>=3:
        featurelosspt = None
        for i in range(len(net.eta)):
            for j in range(1,len(net.pt)-1):
                cuts_ij   = net.nets[i  ][j  ].get_cuts()
                cuts_ijm1 = net.nets[i  ][j-1].get_cuts()
                cuts_ijp1 = net.nets[i  ][j+1].get_cuts()
                flpt = None
                
                cutrange_pt           =  cuts_ijp1-cuts_ijm1
                mean_pt               = (cuts_ijp1+cuts_ijm1)/2.
                distance_from_mean_pt = (cuts_ij  -mean_pt)
                exponent=2.  
                flpt=(distance_from_mean_pt**exponent)/((cutrange_pt**exponent)+0.1) 
                # -----------------------------------------------------
              
                if featurelosspt == None:
                    featurelosspt = flpt
                else:
                    featurelosspt = featurelosspt + flpt

                
        sumptlosses = torch.sum(featurelosspt)/features
        loss.ptloss = epsilon*sumptlosses

    return loss
