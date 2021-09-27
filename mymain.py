import torch
import numpy as np
from mySOM import SOM
from myRBF import RBF
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn import svm
device = torch.device('cpu')

data_train = scio.loadmat('data_train.mat')
data_label = scio.loadmat('label_train.mat')
data_test = scio.loadmat('data_test.mat')
inputs = torch.tensor(data_train['data_train']).to(device)
labels = torch.tensor(data_label['label_train']).to(device)
tests = torch.tensor(data_test['data_test']).to(device)


def SOM_RBF(no_neuron):

    print('Building SOM model...')

    model = SOM(input_size=inputs.size()[1], neuron_size=no_neuron)
    model = model.to(device)

    epoches = list()
    losses = list()

    for epoch in range(1000):
        running_loss = 0

        loss = model.self_org(inputs, epoch)
        running_loss += loss

        epoches.append(epoch)
        losses.append(running_loss)
        # if (epoch+1)%100 == 0:
        #     print('self-org phrase epoch = %d, loss = %.2f'%(epoch+1, running_loss))

    for epoch in range(500*100):
        running_loss = 0

        loss = model.conver(inputs)
        running_loss += loss
        epoches.append(epoch+1000)
        losses.append(running_loss)
        # if (epoch+1)%1000 == 0:
        #     print('converge phrase epoch = %d, loss = %.2f'%(epoch+1, running_loss))

    center_vc = model.weight.T
    print('Buliding RBF model...')

    model = RBF(input_size=inputs.size()[
                1], center_vc=center_vc, output_size=1)
    model = model.to(device)

    model.train(inputs, labels)
    pred1 = model.test(test_input=inputs).unsqueeze(1)
    #accuracy1 = (pred1.to(device) == labels).sum()/torch.tensor(inputs.size(0)).float()
    accuracy1 = (pred1 == labels).sum()/torch.tensor(inputs.size(0)).float()
    pred_test = model.test(tests)
    print('%d neurons SOM+RBF accuracy is %.4f' % (no_neuron, accuracy1))

    plt.plot(epoches, losses, label=u'%d neurons' % no_neuron)
    plt.legend()
    print("%d neurons prediction result:"%no_neuron)
    print(pred_test)
    return pred_test


def SVM():
    X = data_train['data_train']
    y = np.squeeze(data_label['label_train'])
    test = data_test['data_test']
    print('Building SVM model')
    clf = svm.SVC()
    clf.fit(X,y)
    predictions = clf.predict(X)
    accuracy = (predictions == y).sum().__float__() / X.shape[0]
    predictions_y = clf.predict(test)

    print('SVM accuracy is %.4f' % accuracy)
    return predictions_y


if __name__ == '__main__':
    SOM_RBF(4)
    SOM_RBF(9)
    SOM_RBF(16)
    SOM_RBF(25)
    SOM_RBF(36)
    SOM_RBF(49)
    SOM_RBF(64)
    SOM_RBF(81)
    SOM_RBF(100)

    SVM_pred = SVM()
    print("SVM prediction result:")
    print(SVM_pred)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Compare loss for different neurons in training')
    plt.savefig('./loss.eps')
    plt.show()

