import os
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from mydata import myowndataset
from mygnngraphrepr import mygnnGraphRepr
from mynodeembedding import mynodeEmbedding
from torch.utils.tensorboard import SummaryWriter

###模型参数定义

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Classification with GNN')
    parser.add_argument('--task_name', type=str, default='不限幅混合',
                        help='task name')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--graph_pooling', type=str, default='set2set',
                        help='graph pooling strategy mean or sum (default: set2set)')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='dimensionality of hidden units in GNNs (default: 32)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--save_test', action='store_true')#运行时该变量有传参就将该变量设为True
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=25,
                        help='early stop (default: 10)')#Early Stop的概念非常简单，在我们一般训练中，经常由于过拟合导致在训练集上的效果好，而在测试集上的效果非常差。因此我们可以让训练提前停止，在测试集上达到最好的效果时候就停止训练，而不是等到在训练集上饱和在停止，这个操作就叫做Early Stop
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 4)')#num_worker好处是寻batch快
    parser.add_argument('--dataset_root', type=str, default="toy",
                        help='dataset root')
    args = parser.parse_args()#解析参数
    return args


def prepartion(args):
    save_dir = os.path.join('saves', args.task_name)#连接两个或更多的路径名组件
    if os.path.exists(save_dir):
        for idx in range(1000):#？
            if not os.path.exists(save_dir + '=' + str(idx)):#str() 函数将对象转化为适于人阅读的形式
                save_dir = save_dir + '=' + str(idx)
                break

    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)#用来创建多层目录,exist_ok：是否在目录存在时触发异常。如果exist_ok为False（默认值），则在目标目录已存在的情况下触发FileExistsError异常；如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常。
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)

    ####定义训练函数
def train(model, device, loader, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        y_pred = model.forward(batch)
        y = batch.y.view(-1,).long()
        optimizer.zero_grad()#梯度置零
        loss = criterion_fn(y_pred, y)#交叉低损失函数
        loss.backward()#反向传播，计算当前梯度；
        optimizer.step()#根据梯度更新网络参数
        loss_accum += loss.detach().cpu().item()#返回损失值
    return loss_accum / (step + 1)

###定义验证函数
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
    pre = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    F1 = f1_score(y_true, y_pred, average='weighted')
    return acc, pre, recall, F1

###定义测试函数

def test(model, device, loader):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model.predict(batch).view(-1,)
            y_pred.append(pred.detach().cpu())
            y_true.append(batch.y.view(-1, ).detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    acc = accuracy_score(y_true, y_pred)
    return y_pred,y_true,acc

###主函数

def main(args):
    ###导入参数
    prepartion(args)
    nn_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }
    # nn_params2 = {
    #     'num_layers': args.num_layers,
    #     'emb_dim': args.emb_dim,
    #     'drop_ratio': args.drop_ratio,
    #
    # }


    ####导入数据
    dataset = myowndataset(save_root=args.dataset_root)
    ###划分训练集、验证集、测试集
    train_data = dataset[:1440]
    valid_data = dataset[1440:2160]
    test_data = dataset[2160:]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ####定义损失函数
    criterion_fn = torch.nn.CrossEntropyLoss()

    device = args.device
    model = mygnnGraphRepr(**nn_params).to(device)
    # model2= mynodeEmbedding(**nn_params2).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}', file=args.output_file, flush=True)
    print(model, file=args.output_file, flush=True)

    # print(f'#Params: {num_params}')
    # print(model)
    ###定义优化器和学习率调整
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)#等间隔调整学习率

    # writer = SummaryWriter(log_dir=args.save_dir)#可视化网络结构和参数

###开始训练
    not_improved = 0
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch), file=args.output_file, flush=True)
        print('Training...', file=args.output_file, flush=True)

        training = train(model, device, train_loader, optimizer, criterion_fn)

        print('Evaluating...', file=args.output_file, flush=True)

        validing= eval(model, device, valid_loader)

        print({'Train': training, 'Validation': validing}, file=args.output_file, flush=True)

        acc= eval(model, device, valid_loader)[0]

        # writer.add_scalar('valid', validing, epoch)
        # writer.add_scalar('train', training, epoch)
        ####准确率最高时保存模型
        if acc > best_acc:
            best_acc = acc
            if True:
                print('Saving model...', file=args.output_file, flush=True)

                # checkpoint = {
                #     'epoch': epoch, 'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_acc,
                #     'num_params': num_params
                # }
                # torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pt'))

                torch.save(model, '不限幅混合-1.pt')
                # torch.save(model2,'sw.pt')
                ###测试集测试
                print('Predicting on test data...', file=args.output_file, flush=True)
                y_pred = test(model, device, test_loader)
                print('Saving test submission file...', file=args.output_file, flush=True)
                print({'y_pred': y_pred[0],'y_true':y_pred[1],'ACC':y_pred[2]},file=args.output_file, flush=True)

            not_improved = 0
            ###early_stop设计
        else:
            not_improved += 1
            if not_improved == args.early_stop:
                print(f"Have not improved for {not_improved} epoches.", file=args.output_file, flush=True)
                break

        scheduler.step()
        print(f'Best validation acc so far: {best_acc}', file=args.output_file, flush=True)

    # writer.close()

    args.output_file.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)




















