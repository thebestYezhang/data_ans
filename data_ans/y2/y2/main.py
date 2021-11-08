import torch
import numpy as np
from argparse import ArgumentParser
import random
from torch.nn import init
from model import TotalNet
import dataloader
import os
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchvision import transforms


os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 指定GPU


if __name__ == '__main__':
    parser = ArgumentParser("预测")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)


    args = parser.parse_args()


    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("seed:", seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = dataloader.DataLoader(batch_size=args.batch_size, istrain=True)
    test_loader = dataloader.DataLoader(istrain=False)
    train_data = train_loader.get_data()
    test_data = test_loader.get_data()

    # 模型初始化
    model = TotalNet().to(device)
    # model.apply(weights_init_xavier)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9, last_epoch=-1)

    print('training ... ')
    mse_lst = []
    for epoch in range(args.epochs):
        # train
        model.train()
        LOSS = 0
        for i, (im, label) in enumerate(train_data):
            im = im.type(torch.FloatTensor)
            im = im.to(device)
            label = label.type(torch.FloatTensor)
            label = label.to(device)


            optimizer.zero_grad()
            output = model(im)
            loss = F.mse_loss(output, label).to(device)
            # temp_tensor = output[0, :, : ,:]
            # temp_tensor = temp_tensor.squeeze(0).squeeze(0)
            # new_img_PIL = transforms.ToPILImage()(temp_tensor)
            # new_img_PIL.save("a.tif")


            loss.backward()
            optimizer.step()
            LOSS = LOSS + loss.item()
        train_loss = LOSS / (i + 1)
        print("train_loss", train_loss)

        # test
        model.eval()
        MSE = 0.0
        with torch.no_grad():
            for i, (im, label) in enumerate(test_data):
                im = im.type(torch.FloatTensor)
                im = im.to(device)
                label = label.type(torch.FloatTensor)
                label = label.to(device)

                output = model(im)
                mse = F.mse_loss(output, label)
                MSE = MSE + mse.item()
        average_mse = MSE / (i + 1)
        mse_lst.append(average_mse)
        print("Epoch {} Test Results: loss={:.3f} MSE={:.3f}".format(epoch, train_loss, average_mse))

    plt.plot(list(range(1, args.epochs + 1)), mse_lst, "b")
    plt.legend(['MSE'], loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.savefig('mse.jpg')









