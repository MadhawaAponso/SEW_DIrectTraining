import os
import time
import datetime
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate

from spikingjelly.clock_driven import layer, functional

import utils
import smodels

def parse_args():
    class Args:
        model = "SEWResNet"
        train_data_path = "/content/drive/MyDrive/npz_events_best"
        test_data_path = "/content/drive/MyDrive/npz_events_test_best"
        device = "cuda"
        batch_size = 16
        epochs = 25
        workers = 4
        lr = 5e-4
        momentum = 0.9
        weight_decay = 1e-4
        lr_step_size = 64
        lr_gamma = 0.1
        print_freq = 64
        output_dir = "/content/drive/MyDrive/logs"
        resume = ""
        start_epoch = 0
        sync_bn = False
        test_only = False
        amp = True
        world_size = 1
        dist_url = "env://"
        tb = True
        adam = True
        connect_f = "ADD"
        T_train = 12

    return Args()

_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SpikingjellyDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.files = []  # Store (file_path, label) tuples

        # Iterate through each folder (gesture class)
        for class_label, class_folder in enumerate(sorted(os.listdir(dataset_path))):
            class_path = os.path.join(dataset_path, class_folder)
            if os.path.isdir(class_path):  # Ensure it's a directory
                for file in sorted(os.listdir(class_path)):
                    if file.endswith(".npz"):  # Only use .npz files
                        self.files.append((os.path.join(class_path, file), class_label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]  # Get file path and label
        data = np.load(file_path, allow_pickle=True)

        # Ensure keys exist
        required_keys = {"x", "y", "t", "p", "f"}
        if not required_keys.issubset(data.files):
            raise ValueError(f"Missing keys in {file_path}: {set(data.files) - required_keys}")

        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        t = data["t"].astype(np.float32)
        p = data["p"].astype(np.float32)
        folder_name = data["f"].item()

        events = np.stack([x, y, t, p], axis=1)  # Shape: (num_events, 4)

        return torch.from_numpy(events), label, folder_name

# Loader Class for Batch Processing
class Loader:
    def __init__(self, dataset, args, device, distributed, batch_size, drop_last=True , to_train = True):
        self.device = device
        if distributed is True:
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            if to_train:
                self.sampler = torch.utils.data.RandomSampler(dataset)
            else:self.sampler = torch.utils.data.SequentialSampler(dataset)

        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=self.sampler,
                                                  num_workers=args.workers, pin_memory=True,
                                                  collate_fn=collate_events, drop_last=drop_last)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)

# Collate function to handle batching of events
def collate_events(data):
    labels = []
    events = []
    folder_names = []
    for i, d in enumerate(data):
        labels.append(d[1])
        folder_names.append(d[2])
        ev = torch.cat([d[0], i * torch.ones((len(d[0]), 1), dtype=torch.float32)], 1)
        events.append(ev)
    events = torch.cat(events, 0)
    labels = default_collate(labels)
    folder_names = default_collate(folder_names)
    return events, labels, folder_names

class QuantizationLayerVoxGrid(nn.Module):
    def __init__(self, dim, mode):
        nn.Module.__init__(self)
        self.dim = dim
        self.mode = mode

    def forward(self, events):
        epsilon = 10e-3
        B = int(1+events[-1, -1].item())
        # tqdm.write(str(B))
        num_voxels = int(2 * np.prod(self.dim) * B)
        C, H, W = self.dim
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        # get values for each channel
        x, y, t, p, b = events.T
        x = x.to(torch.int64)
        y = y.to(torch.int64)
        p = p.to(torch.int64)
        b = b.to(torch.int64)
        # normalizing timestamps
        unit_len = []
        t_idx = []
        for bi in range(B):
            bi_idx = events[:, -1] == bi
            t[bi_idx] /= t[bi_idx].max()
            unit_len.append(int(bi_idx.float().sum() / C))
            _, t_idx_ = torch.sort(t[events[:, -1] == bi])
            t_idx.append(t_idx_)
        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b


        for i_bin in range(C):
            values = torch.zeros_like(t)
            for bi in range(B):
                bin_idx = t_idx[bi][i_bin * unit_len[bi]: (i_bin + 1) * unit_len[bi]]
                bin_values = values[events[:, -1] == bi]
                bin_values[bin_idx] = 1
                values[events[:, -1] == bi] = bin_values
            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)#.clamp(0, 1)
        if self.mode == "TB":
            vox = vox.permute(2, 0, 1, 3, 4).contiguous()
        elif self.mode == "BT":
            vox = vox.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise Exception
        return vox

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, quantizer,print_freq, scaler=None, T_train=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for event, target , foldername in data_loader:
        start_time = time.time()
        image = quantizer(event)
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



def evaluate(model, criterion, data_loader, device,quantizer, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for event, target , foldername in data_loader:
            image = quantizer(event)
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    # print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def main(args):

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.


    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}')

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    output_dir += f'_steplr{args.lr_step_size}_{args.lr_gamma}'

    if args.adam:
        output_dir += '_adam'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)



    device = torch.device(args.device)

    # data_path = args.data_path
    dataset_train = SpikingjellyDataset(args.train_data_path)
    dataset_test = SpikingjellyDataset(args.test_data_path)

    print(f"dataset_train {len(dataset_train)} , dataset_test {len(dataset_test)}")

    distributed = False
    batch_size  = 16
    data_loader = Loader(dataset=dataset_train, args=args, device=device, distributed=distributed, batch_size=batch_size)
    data_loader_test = Loader(dataset=dataset_test, args=args, device=device, distributed=distributed, batch_size=batch_size , to_train = False)

    quantizer = QuantizationLayerVoxGrid(dim=(16, 128 ,128), mode="BT")

    model = smodels.SEWResNet(args.connect_f)
    print("Creating model")

    model.to(device)

    num_params = count_trainable_params(model)
    print(f"Total Trainable Parameters: {num_params:,}")
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    if args.adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.amp:
        scaler = torch.amp.GradScaler() if args.amp else None
    else:
        scaler = None

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model_without_ddp.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     args.start_epoch = checkpoint['epoch'] + 1
    #     max_test_acc1 = checkpoint['max_test_acc1']
    #     test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']


    # if args.tb and is_main_process():
    #     purge_step_train = args.start_epoch
    #     purge_step_te = args.start_epoch
    #     train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
    #     te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
    #     with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
    #         args_txt.write(str(args))

    #     print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            utils.train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,quantizer, args.print_freq, scaler, args.T_train )
        # print(f"Train Loss {train_loss} , Train_acc1 = {train_acc1} , Train acc5 = {train_acc5}")
        # if is_main_process():
        #     train_tb_writer.add_scalar('train_loss', train_loss, epoch)
        #     train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
        #     train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test,quantizer=quantizer, device=device, header='Test:')
        print(f"Epoch {epoch} ,Train Loss {train_loss} , Train_acc1 = {train_acc1} , Train acc5 = {train_acc5} , test_loss = {test_loss} , test_acc_1 ={test_acc1}")
        # if te_tb_writer is not None:
        #     if is_main_process():

        #         te_tb_writer.add_scalar('test_loss', test_loss, epoch)
        #         te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
        #         te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


        # if max_test_acc1 < test_acc1:
        #     max_test_acc1 = test_acc1
        #     test_acc5_at_max_test_acc1 = test_acc5
        #     save_max = True


        # if output_dir:

        #     checkpoint = {
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args,
        #         'max_test_acc1': max_test_acc1,
        #         'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
        #     }

            # if save_max:
            #     save_on_master(
            #         checkpoint,
            #         os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        # print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        # print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        # print(output_dir)
    # if output_dir:
    #     save_on_master(
    #         checkpoint,
    #         os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1



if __name__ == "__main__":
    args = parse_args()
    main(args)

