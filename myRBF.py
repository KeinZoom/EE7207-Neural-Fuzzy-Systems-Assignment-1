import torch
import torch.nn as nn
import numpy as np
device = torch.device('cpu')
class RBF(nn.Module):
    def __init__(self, input_size, center_vc, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.center_vc = center_vc

    def max_dis(self):
        maxdis = 0
        for i in self.center_vc:
            distance = nn.PairwiseDistance(p=2)(i, self.center_vc)
            if distance.max()>maxdis:
                maxdis = distance.max()
        return maxdis

    def train(self, inputs, label):

        maxdis = self.max_dis()
        sigma = maxdis/np.sqrt( 2 * self.center_vc.size(0))

        x = inputs.unsqueeze(1)
        center = self.center_vc.unsqueeze(0)

        distance = (x-center).pow(2).sum(2)
        bias_para = torch.ones_like(label)
        Gussian_fn = torch.exp(-distance / (2 * sigma **2))
        Gussian_fn = torch.cat((Gussian_fn,bias_para),1)

        #weight
        self.weight = torch.matmul(
            torch.matmul(
                torch.inverse(
                    torch.matmul(Gussian_fn.T, Gussian_fn)), Gussian_fn.T),label.double())

    
    def test(self, test_input):
        maxdis = self.max_dis()
        sigma = maxdis / np.sqrt( 2 * self.center_vc.size(0))

        x = test_input.unsqueeze(1)
        center = self.center_vc.unsqueeze(0)

        distance = (x-center).pow(2).sum(2)

        bias_para = torch.ones(test_input.size(0),1)

        Gussian_fn = torch.exp(-distance / (2 * sigma **2))
        #Gussian_fn = torch.cat((Gussian_fn,bias_para.to(device)),1)
        Gussian_fn = torch.cat((Gussian_fn,bias_para),1)
        Y = torch.matmul(Gussian_fn, self.weight)
        
        pred = torch.tensor([1 if i>0 else -1 for i in Y])
        return pred