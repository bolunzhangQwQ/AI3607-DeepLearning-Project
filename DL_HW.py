#coding=gbk
import time
from jittor.lr_scheduler import CosineAnnealingLR
import numpy as np
import jittor as jt
if jt.has_cuda:
    jt.flags.use_cuda = 1
import random
import jittor.nn as nn
from jittor.dataset.cifar import CIFAR10
import matplotlib.pyplot as plt
 
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm(32), # 添加批量归一化层
            nn.Pool(kernel_size=2, stride=2, op='maximum'),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm(64), # 添加批量归一化层
            nn.Pool(kernel_size=2, stride=2, op='maximum'),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm(128), # 添加批量归一化层
            nn.Dropout(p=0.2),
            nn.Conv(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm(256), # 添加批量归一化层
            nn.Dropout(p=0.2),
            nn.Conv(256, 256, kernel_size=3, padding=1),
            nn.Pool(kernel_size=2, stride=2, op='maximum'),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 2048),
            nn.BatchNorm(2048), # 添加批量归一化层
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 1024),
            nn.BatchNorm(1024), # 添加批量归一化层
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

    def execute(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
class DeepPermNet(nn.Module):
    def __init__(self):
        super(DeepPermNet, self).__init__()
        self.Alexnet = AlexNet()
        self.batch_norm = nn.BatchNorm1d(1024 * 4)
        self.fully_connected = nn.Sequential(
            nn.Linear((1024 * 4), 8192), 
            nn.BatchNorm1d(8192),
            nn.ReLU(), 
            nn.Dropout(p=0.1),
            nn.Linear(8192, 4096), 
            nn.BatchNorm1d(4096),
            nn.ReLU(), 
            nn.Dropout(p=0.1),
            nn.Linear(4096, 16)
        )
        self.l2_regularization = nn.Linear(16, 16, bias=False)

    def execute(self, input1, input2, input3, input4):
        input1, input2, input3, input4 = self.Alexnet(input1), self.Alexnet(input2), self.Alexnet(input3), self.Alexnet(input4)
        concatenated_input = jt.concat([input1, input2, input3, input4], dim=1)
        normalized_input = self.batch_norm(concatenated_input)
        output = self.fully_connected(normalized_input)
        output = self.l2_regularization(output)
        reshaped_output = output.view(-1, 4, 4)
        return reshaped_output
def spliter(images):
    split_arrays = [np.array([]) for _ in range(4)]
    split_targets = []
    for idx, image in enumerate(images):
        zero_matrix = np.zeros((1, 4, 4))
        shuffled_indices = list(range(4))
        random.shuffle(shuffled_indices)
        img = jt.unsqueeze(jt.float32(image), 0)
        img_parts = [img[:, :, 0:16, 0:16], img[:, :, 0:16, 16:32], img[:, :, 16:32, 0:16], img[:, :, 16:32, 16:32]]
        for i, j in enumerate(shuffled_indices):
            zero_matrix[0][j][i] = 1
            if idx == 0:
                split_arrays[j] = img_parts[i]
            else:
                split_arrays[j] = jt.contrib.concat([split_arrays[j], img_parts[i]], dim=0)
        if idx == 0:
            split_targets = zero_matrix
        else:
            split_targets = jt.contrib.concat([split_targets, zero_matrix], dim=0)
    split_targets = jt.array(split_targets)
    return tuple(jt.array(arr) for arr in split_arrays) + (split_targets,)
    
def Validate(model, batch_size, test_list):
    model.eval()
    total_acc = 0
    total_num =0
    for index, (inputs, targets) in enumerate(test_list):
        image1, image2, image3, image4, labels = spliter(jt.float32(inputs.transpose((0, 3, 1, 2))))
        outputs = model(image1, image2, image3, image4)
        outputs = jt.argmax(outputs.view(batch_size * 4, 4), dim=1)[0].view(batch_size, 4)
        labels = jt.nonzero(labels.view(batch_size * 4, 4))[:, 1].view(batch_size, 4)
        correct = jt.equal(outputs, labels).all(dim=1).sum().item()
        total_acc += correct
        total_num += batch_size
    test_acc = total_acc / total_num
    print('Acc =', test_acc)
    return test_acc

def train(model, epoch, optimizer, train_list, loss_list,epoches=1, batch_size=100):
    loss_=[]
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_list):
        image1, image2, image3, image4, labels=spliter(jt.float32(inputs.transpose((0,3,1,2))))
        outputs=model(image1, image2, image3, image4)
        outputs=outputs.reshape((batch_size*4,4))
        labels=labels.reshape((batch_size*4,4))
        labels=jt.nonzero(labels)
        labels=labels[:,1]
        loss = nn.cross_entropy_loss(outputs, labels) 
        loss_.append(loss)                                      
        print('Train Epoch: {} of {} [{}/{}] Loss: {:.6f}'.format(epoch+1, epoches, batch_idx * batch_size, 
                len(train_list), loss))
        optimizer.step(loss)
    loss_list.append(np.mean(loss_))

def main():
    learning_rate = 0.1
    loss_list=[]
    accs=[]
    max=0
    epoches=40
    momentum = 0.9
    weight_decay = 1e-4
    #Image spliter
    batch_size = 100
    train_list = CIFAR10(train=True)
    test_list  = CIFAR10(train=False)
    train_list = train_list.set_attrs(batch_size=batch_size)
    test_list  = test_list.set_attrs(batch_size=batch_size)
    #Train
    model = DeepPermNet()
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoches, eta_min=1e-8)
    for epoch in range(epoches):
        train(model,epoch, optimizer, train_list, loss_list,epoches)
        acc=Validate(model=model,batch_size=batch_size,test_list=test_list)
        accs.append(acc)
        if max<=acc:
            max=acc
        scheduler.step()
    #showacc
    plt.figure()
    plt.plot(range(len(accs)),accs)
    plt.title("Acc")
    plt.show()
    print(f"Best acc:{max}")
    plt.figure()
    plt.plot(range(len(loss_list)),loss_list)
    plt.title("Loss")
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time() # 记录程序结束时间
    print("程序运行时间为：{:.2f}秒".format(end_time - start_time))
