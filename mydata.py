import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from  torch_geometric.data import DataLoader
import os
import xlrd

def mydataset(num_nodes, num_node_features, num_edges,num_edge_features):

    file = xlrd.open_workbook(os.path.join('D:\LZY\小论文\甄九宝毕业资料打包\甄九宝毕业资料打包\第三章\MPNN\dataset','B-01-BXF.xlsx'))
    table = file.sheets()[0]#选取要读的sheet表单
    nrows = table.nrows
    result=[]
    for i in range(nrows):
        if i == -1:
            continue
        mydata = table.row_values(i)#获取整行的值（数组）
        myedge_index = mydata[0:2*num_edges]
        mynum_node_features = mydata[2*num_edges:12*num_nodes+2*num_edges]
        mynum_edge_features = mydata[12*num_nodes+2*num_edges:3*num_edges+12*num_nodes+2*num_edges]
        mylabels = mydata[3*num_edges+12*num_nodes+2*num_edges:]
        edge_index = np.reshape(myedge_index, (2,num_edges), order='C')
        node_features = np.reshape(mynum_node_features, (num_nodes,num_node_features), order='C')
        edge_features = np.reshape(mynum_edge_features, (num_edges,num_edge_features), order='C')
        labels = np.reshape(mylabels, (1,1), order='C')
        edge_index = torch.tensor(edge_index ,dtype=torch.long)
        x = torch.tensor(node_features,dtype=torch.float)
        edge_attr =torch.tensor(edge_features,dtype=torch.float)
        y = torch.tensor(labels,dtype=torch.float)
        data = Data(edge_index=edge_index,x=x,edge_attr=edge_attr,y=y )
        result.append(data)
    return result

# In Memory Dataset
class myowndataset(InMemoryDataset):
    def __init__(self, save_root, transform=None, pre_transform=None):
        """
        :param save_root:保存数据的目录
        :param pre_transform:在读取数据之前做一个数据预处理的操作
        :param transform:在访问之前动态转换数据对象(因此最好用于数据扩充)
        """
        super(myowndataset, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):  # 原始数据文件夹存放位置,这个例子中是随机出创建的，所以这个文件夹为空
        return ['origin_dataset']

    @property
    def processed_file_names(self):
        return ['mydataset.pt']

    def download(self):  # 这个例子中不是从网上下载的，所以这个函数pass掉
        pass

    def process(self):   # 处理数据的函数,最关键（怎么创建，怎么保存）

        data_list = mydataset(num_nodes=9, num_node_features=12, num_edges=18,num_edge_features=3)
        data_save, data_slices = self.collate(data_list) # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
        torch.save((data_save, data_slices), self.processed_file_names[0])






# if __name__ == "__main__":
#     dataset = myowndataset(save_root="toy")  # 样本（图）
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=True,num_workers=4) # batch_size=5实现了平行化——就是把5张图放一起了
# #
#     for batch in data_loader: # 循环
#         pass
















