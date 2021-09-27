
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cpu')

class SOM(nn.Module):
    def __init__(self, input_size, neuron_size, lr=0.1, sigma=None):
        super(SOM, self).__init__()

        self.input_size = input_size
        self.neuron_size = neuron_size
        self.lr = lr
        self.weight = nn.Parameter(torch.randn( input_size, neuron_size),requires_grad=False)
        self.location = nn.Parameter(torch.Tensor(list(self.gen_location())),requires_grad=False)
        self.sigma = (self.location[-1]-self.location[0]).pow(2).sum().pow(0.5).div(2)

        

    def gen_location(self):

        for x in range( int(pow(self.neuron_size, 0.5))):
            for y in range( int ( pow (self.neuron_size, 0.5))):
                yield( x, y)
    
    def forward(self, input):
        data_size = input.size()[0]
        input = input.view( data_size, -1, 1)
        weight = self.weight

        distances = nn.PairwiseDistance( p = 2)( input, weight)

        # neuron with minimum distance
        loss, mdn = distances.min( dim = 1, keepdim = True)
        mdn_location = self.location[mdn]

        return loss.sum().div_(data_size).item(), mdn_location

    def self_org(self, input, iteration):
        data_size = input.size()[0]
        #lr
        lr = self.lr * np.exp(-iteration/1000)

        #neighbourhood
        time_const = 1000/np.log(self.sigma)
        sigma = self.sigma * np.exp(-iteration/time_const)
        sigma_sq = pow(sigma, 2)
        #neibour function
        loss, mdn_location = self.forward( input )

        dis_sq = self.location.float () - mdn_location.float()
        dis_sq.pow_(2)
        dis_sq = torch.sum(dis_sq, dim = 2)
        h = np.exp(-dis_sq/2/sigma_sq)
        #h = np.exp(-dis_sq.cpu()/2/sigma_sq).to(device)
        h.unsqueeze_(1)
        #delta
        delta = lr * h * ( input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0) / data_size
        self.weight += delta

        return loss

    def conver(self, input):
        data_size = input.size()[0]
        #lr
        lr = 0.01

        #neighbourhood
        loss, mdn_location = self.forward(input)
        dis_sq = self.location.float() - mdn_location.float()
        dis_sq.pow_(2)
        dis_sq = torch.sum(dis_sq, dim = 2)

        h = torch.zeros_like(dis_sq)
        h[torch.where(dis_sq==0)] = 1
        h.unsqueeze_(1)

        #delta
        delta = lr * h * ( input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0) / data_size
        self.weight += delta

        return loss
