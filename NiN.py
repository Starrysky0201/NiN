"""
网络模型：NiN
图像数据集：CIFAR-10
图片大小：32x32
图片通道：rgb
数据集地址：http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz
"""
# 导包
import os.path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
# 定义数据集的路径为当前文件夹
data_path = './cifar-10-batches-py'

# 定义图片转张量，并进行归一化处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5))
    ]
)
# 将读取的数据的像素值转换为rgb图片
def toImage(val):
    """
    :param val: 【3,32,32】
    分别为rgb三个通道每个通道为32*32个值
    需要将它转置为：【32,32,3】
    为每个点的rgb值
    :return: Image
    """
    x = 32
    y = 32
    val = np.reshape(val, (3, 32, 32))
    val = val.T
    im = Image.fromarray(np.uint8(val))
    return im
# 读取batches.meta的标签字典
def get_label_names(file_path):
    """
    根据官方文档batches.meta中有10个标签，利用pickle读取
    :param file_path 文件路径
    :return: 标签字典
    """
    batche_path = os.path.join(file_path, 'batches.meta')
    with open(batche_path, 'rb',) as f:
        label_names = pickle.load(f, encoding='bytes')
        label_names = [val.decode(encoding = 'utf-8') for val in label_names[b'label_names']]
        return label_names
# 获取标签
label_names = get_label_names(data_path)


def unpickle(file):
    """
    官方提供的函数，读取一个batch，一共有五个
    :param file:
    :return:
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CifarDataset(Dataset):
    """
    定义读取Cifar数据集的类，分别对train和test做处理
    """
    def __init__(self, data_path, transform,mode = 'train'):
        self.Data = []
        self.Label = []
        self.transform = transform
        if mode == 'train':
            data_files = ['data_batch_' + str(i) for  i in range(1,5+1)]
            self.file_data_paths = [os.path.join(data_path, data_file) for data_file in data_files]
        else:
            self.file_data_paths = [os.path.join(data_path, 'test_batch')]
        for file_path in self.file_data_paths:
            dict = unpickle(file_path)
            self.Data.extend([toImage(val) for val in dict[b'data']])
            self.Label.extend(dict[b'labels'])
    def __getitem__(self, item):
        data = self.transform(self.Data[item])
        label = self.Label[item]
        return data, label
    def __len__(self):
        return len(self.Label)

# 构造一个NiN块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """
    这个是看书上的
    """
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )
# 定义NiN网络模型
net = nn.Sequential(
                                                                                    # [1, 3, 32, 32]
    nin_block(3, 96, kernel_size=11, strides=2, padding= 1),  # [1, 96, 12, 12]
    nn.MaxPool2d(3, 1),                                             # [1, 96, 10, 10]
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),  # [1, 256, 10, 10]
    nn.MaxPool2d(3, 1),                                             # [1, 256, 8, 8]
    nin_block(256, 384,kernel_size=3, strides=1, padding=1),  # [1, 384, 8, 8]
    nn.MaxPool2d(3,2),                                              # [1,384, 3, 3]
    nn.Dropout(0.5), # 避免模型过拟合
    nin_block(384, 10, kernel_size=3, strides=1, padding=1), #[1, 10, 3, 3]
    nn.AdaptiveAvgPool2d((1,1)), # [1, 10, 1, 1]
    nn.Flatten() # [1, 10] # 展平
)
# 定义批大小
batch_size = 100

# 获取训练和测试数据以及它们的批大小
traindataset = CifarDataset(data_path, transform,'train')
testdataset = CifarDataset(data_path, transform,'test' )
traindataloader = DataLoader(traindataset, batch_size=batch_size,shuffle=True)
testdataloader = DataLoader(testdataset, batch_size = batch_size, shuffle=True)

# 利用GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义net并将net放入GPU中
net = net.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

# 定义损失和准确率
train_loss = []
test_accuracy = []

# 可视化损失和准确率
def draw_loss_acc(epochs):
    plt.figure()
    plt.plot(range(1, epochs+1),train_loss, label = 'Train loss',color = 'r')
    plt.plot(range(1, epochs+1), test_accuracy, label = 'Test Accuracy',color = 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show(block =False)

# 测试结果的可视化
def predict_image():
    net.eval()
    predict_imgs = []
    predict_labs = []
    # 获取测试集的前10个数据
    for idx in range(0,10):
        img, label = testdataset[idx]
        predict_imgs.append(img)
        predict_labs.append(label)

    predict_imgs = torch.stack(predict_imgs)
    predict_labs = torch.IntTensor(predict_labs)

    with torch.no_grad():
        inputs, labels = predict_imgs.to(device), predict_labs.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        plt.figure()
        for idx, img in enumerate(predict_imgs):
            plt.subplot(2,5,idx+1)
            plt.imshow(testdataset.Data[idx])
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('pre: {}'.format(label_names[predicted[idx]]))
            plt.title('{}'.format(label_names[predict_labs[idx]]))
        plt.show()

# 定义训练函数，并打印损失，这个就比较模版了
def train(epoch):
    net.train()
    running_loss = 0.0
    for inputs, labels in traindataloader:
        # 将数据放入GPU中
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() #梯度清零
        outputs = net(inputs) # 获取输出
        loss = criterion(outputs, labels) # 求损失
        loss.backward() # 反向传播
        optimizer.step() # 参数更新
        running_loss += loss.item() * inputs.size(0)
    # 求平均损失
    train_loss.append(running_loss/len(traindataloader.dataset))
    print("Epoch {}, Loss: {:.4f}".format(epoch+1,running_loss/len(traindataloader.dataset)))

# 定义测试函数，并打印准确率
def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testdataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1) # 获取最大的为预测值
            total +=labels.size(0)
            correct += (predicted == labels).sum().item() # 获取正确的个数
    test_accuracy.append(correct/total)
    print('Epoch {}:Accuracy : {:.2f} % '.format( epoch+1, 100. * correct/total))

if __name__ == '__main__':
    epochs = 3
    for epoch in range(0,epochs):
        train(epoch)
        test(epoch)
    draw_loss_acc(epochs)
    predict_image()



