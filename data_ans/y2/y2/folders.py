import xlrd
import numpy as np
import torch.utils.data as data
import torch
import random

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

class Folder(data.Dataset):
    def __init__(self, train):

        self.train = train

        # 打开excel
        wb = xlrd.open_workbook('test.xlsx')
        # 按工作簿定位工作表
        sh = wb.sheet_by_name('Sheet1')

        data_dict = dict()
        train_dict = dict()
        test_dict = dict()
        for title in sh.row_values(0):
            data_dict[title] = []  # 读取第一行的标题, 每个标题作为data_dict的一个键, 初始化各个键的值为空列表
            train_dict[title] = []
            test_dict[title] = []
        index_dict = {0: "x1", 1: "x2", 2: "y1", 3: "y2", 4: "y3"}
        key_list = ["x1", "x2", "y1", "y2", "y3"]

        # 载入数据
        row_num = sh.nrows
        col_num = sh.ncols
        for row_i in range(1, row_num):
            for col_i in range(col_num):
                data_dict[index_dict[col_i]].append(sh.cell(row_i, col_i).value)

        # 数据归一化
        scaled_data_dict = self.data_scale(data_dict, key_list)


        # 划分训练集和测试集
        self.index = list(range(row_num-1))
        random.shuffle(self.index)
        train_index = self.index[0:int(round(0.8 * len(self.index)))]
        test_index = self.index[int(round(0.8 * len(self.index))):len(self.index)]

        for row_i in range(len(self.index)):
            if row_i in test_index:
                for col_i in range(col_num):
                    test_dict[index_dict[col_i]].append(scaled_data_dict[index_dict[col_i]][row_i])
            else:
                for col_i in range(col_num):
                    train_dict[index_dict[col_i]].append(scaled_data_dict[index_dict[col_i]][row_i])


        # 扩充维度
        for key, values in train_dict.items():
            train_dict[key] = np.expand_dims(np.array(values), axis=1)
        for key, values in test_dict.items():
            test_dict[key] = np.expand_dims(np.array(values), axis=1)



        input_key = ["x1", "x2"]
        output_key = ["y1", "y2", "y3"]
        if self.train:
            self.input = np.concatenate((train_dict[input_key[0]],train_dict[input_key[1]]),axis=1)
            self.label = np.concatenate((train_dict[output_key[0]],
                                          train_dict[output_key[1]],
                                          train_dict[output_key[2]]), axis=1)
            self.input = torch.from_numpy(self.input)
            self.label = torch.from_numpy(self.label)
        else:
            self.input = np.concatenate((test_dict[input_key[0]], test_dict[input_key[1]]), axis=1)
            self.label = np.concatenate((test_dict[output_key[0]],
                                         test_dict[output_key[1]],
                                         test_dict[output_key[2]]), axis=1)
            self.input = torch.from_numpy(self.input)
            self.label = torch.from_numpy(self.label)
        print(self.input.shape[0])
        print(self.label.shape[0])

    def __getitem__(self, item):
        return self.input[item], self.label[item]

    def __len__(self):
        length = self.input.shape[0]
        return length


    def data_scale(self, data, input_info):
        # 初始化最大值和最小值
        # 把最大值设置的非常小，更容易遇到比它大的去更新最大值(因为不清楚各列的最大值到底多大，有的可能10e-5就是最大值了，因此需要把初始最大值设的非常小)
        max_values = np.zeros(5) - 1e10
        # 同理，把最小值设的非常大。
        min_values = np.zeros(5) + 1e10

        # 分别更新每列的最大值和最小值
        for i, key in enumerate(input_info):
            for j in range(len(data[key])):  # 遍历每一列
                # 更新第i列的最大值
                if data[key][j] > max_values[i]:
                    max_values[i] = data[key][j]
                # 更新第i列的最小值
                if data[key][j] < min_values[i]:
                    min_values[i] = data[key][j]

        # # 打印各列的最大最小值
        print(max_values)
        print(min_values)

        # 得到各列的最大最小值后，并应用缩放公式对各列数据进行特征缩放
        for i, key in enumerate(input_info):
            for j in range(len(data[key])):
                data[key][j] = (data[key][j] - min_values[i]) / (max_values[i] - min_values[i])

        return data

if __name__ == "__main__":
    a = Folder(train=True)











# print(sh.nrows)#有效数据行数
# print(sh.ncols)#有效数据列数
# print(sh.cell(0,0).value)#输出第一行第一列的值
# print(sh.row_values(0))#输出第一行的所有值
# #将数据和标题组合成字典
# print(dict(zip(sh.row_values(0),sh.row_values(1))))
# #遍历excel，打印所有数据
# for i in range(sh.nrows):
#     print(sh.row_values(i))


