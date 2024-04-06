import torch
from mydata import myowndataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model.predict(batch).view(-1,)

            y_true.append(batch.y.view(-1,).detach().cpu())

            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    acc = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred, average='weighted')
    return y_pred,y_true,acc,F1

dataset = myowndataset(save_root="toy")
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
device = torch.device("cpu")

model = torch.load('D:\moxing/biyelunwen\mine3\Model\model9.pt')

validing= eval(model, device, train_loader )

print({'y_pred': validing[0], 'y_true': validing[1], 'ACC': validing[2],'f1':validing[3]})






####检测数据集用
# from mydata import myowndataset
# dataset = myowndataset(save_root="toy")
# print(len(dataset))
# data = dataset[1]
# print(data['x'])
# print(data['edge_index'])
# print(data['edge_attr'])
# print(data['y'])