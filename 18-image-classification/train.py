# Train
from efficientnet import EfficientNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse
from dataset import build_data_set
import os
#from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter()


def train(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        print(images.shape)
        # compute output
        output = model(images)  # 模型输出
        loss = criterion(output, target)    # 损失函数
        print('Epoch ', epoch, loss)

        # compute gradient and do SGD step
        optimizer.zero_grad()   # 梯度清零？
        loss.backward() # 反向传播
        optimizer.step() # 更新模型权重？
        #writer.add_scalar('Loss/train', loss, epoch)
    #writer.close()


def main(args):
    args.classes_num = 2  # 修改一下 num_classes，将 EfficientNet 的全连接层修改我们项目对应的类别数，这里的 args.classes_num 为 2（logo 类与 others 类）。
    '''
    这段代码是说，如果 pretrained model 参数为 True，则自动下载并加载 pretrained model 后进行训练，否则是使用随机数初始化网络。
    '''
    if args.pretrained:
        # 加载模型
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.classes_num, advprop=args.advprop)
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = EfficientNet.from_name(args.arch, override_params={'num_classes': args.classes_num})

    criterion = nn.CrossEntropyLoss() # 有GPU的话加上.cuda()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 训练集
    train_dataset = build_data_set(args.image_size, args.train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 迭代训练
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)
        if epoch % args.save_interval == 0:
            if not os.path.exists(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            # 保存模型
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar.epoch_%s' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u"[+]----------------- 图像分类Sample -----------------[+]")
    parser.add_argument('--train-data', default='./data/train', dest='train_data', help='location of train data')
    parser.add_argument('--image-size', default=224, dest='image_size', type=int, help='size of input image')
    parser.add_argument('--batch-size', default=10, dest='batch_size', type=int, help='batch size')
    parser.add_argument('--workers', default=4, dest='num_workers', type=int, help='worders number of Dataloader')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr',  default=0.001,type=float, help='learning rate')
    parser.add_argument('--checkpoint-dir', default='./ckpts/', dest='checkpoint_dir', help='location of checkpoint')
    parser.add_argument('--save-interval', default=1, dest='save_interval', type=int, help='save interval')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay', help='weight decay')
    parser.add_argument('--arch', default='efficientnet-b0', help='arch type of EfficientNet')
    parser.add_argument('--pretrained', default=True, help='learning rate')
    parser.add_argument('--advprop', default=False, help='advprop')
    args = parser.parse_args()
    main(args)
