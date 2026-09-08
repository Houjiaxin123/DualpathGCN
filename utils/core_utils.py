from argparse import Namespace
from collections import OrderedDict
import os
import pickle 

import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from datasets.dataset_generic import save_splits
from models.model_set_mil import *
from models.model_graph_mil import *
from utils.utils import *
import gc

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_cindex, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss
        # score = val_cindex

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'validation', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("test on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    reg_fn = None

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='amil':
        model_dict = {'n_classes': args.n_classes}
        model = MIL_Attention_FC_surv(**model_dict)
    elif args.model_type == 'patchgcn':
        model_dict = {'num_layers': args.num_gcn_layers, 'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes}
        model = PatchGCN_Surv(**model_dict)
    elif args.model_type == 'h2gcn':
        model_dict = {'feat_dim': 1536, 'hidden_dim': 128, 'class_dim': 4, 'edge_agg': args.edge_agg}
        model = H2GCN(**model_dict)
    elif args.model_type == 'CombinedModel':
        model_dict1 = {'num_layers': args.num_gcn_layers, 'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes}
        model_dict2 = {'feat_dim': 1536, 'hidden_dim': 128, 'class_dim': 4, 'edge_agg': args.edge_agg}
        model_1 = PatchGCN_Surv(**model_dict1)
        model_2 = H2GCN(**model_dict2)
        model = CombinedModel(model_1, model_2, edge_agg=args.edge_agg)
    else:
        raise NotImplementedError

    count_params(model)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:2'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
                                    weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    test_loader = get_split_loader(test_split, testing=args.testing, mode=args.mode, batch_size=args.batch_size)
    train_eval_loader = get_split_loader(train_split, testing=args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=5, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc, args.bag_loss)
            stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args.bag_loss)
            if stop:
                break

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur))))
    results_train_dict, train_cindex = summary_survival(model, train_eval_loader, args.n_classes)
    results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)
    results_test_dict, test_cindex = summary_survival(model, test_loader, args.n_classes)
    print('train c-Index: {:.4f}'.format(train_cindex))
    print('Val c-Index: {:.4f}'.format(val_cindex))
    print('test c-Index: {:.4f}'.format(test_cindex))
    writer.close()
    return results_train_dict, train_cindex,results_val_dict, val_cindex,results_test_dict, test_cindex


def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, bag_loss=None):
    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, batch in enumerate(loader):

        if len(batch) == 5:
            data_WSI, label, event_time, c, case_id = batch
        else:
            data_WSI, label, event_time, c = batch

        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 100_000:
                continue


        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)

        out = model(x_path=data_WSI)
        hazards = out["hazards"]
        S = out["S"]
        Y_hat = out["Y_hat"]
        logits = out["logits"]

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg


        loss_value = loss.item()

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, bag_loss=None):
    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, batch in enumerate(loader):

        if len(batch) == 5:
            data_WSI, label, event_time, c, case_id = batch
        else:
            data_WSI, label, event_time, c = batch
        
        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 100_000:
                continue

        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            out = model(x_path=data_WSI)
            hazards = out["hazards"]
            S = out["S"]
            Y_hat = out["Y_hat"]
            logits = out["logits"]

            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)

            loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)
        print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch,val_loss_surv,val_loss,c_index))


    if early_stopping:
        assert results_dir
        early_stopping(epoch, c_index, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_survival(model, loader, n_classes):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    patient_results = {}

    for batch_idx, batch in enumerate(loader):

        # 兼容新旧返回格式
        if len(batch) == 5:
            data_WSI, label, event_time, c, case_id = batch
        else:
            raise ValueError(
                "Dataset must return (data_WSI, label, event_time, c, case_id). "
                "Please modify __getitem__ first."
            )

        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 100_000:
                print(f"[Skip] {case_id}: too many nodes {data_WSI.x.shape[0]}")
                continue

        data_WSI = data_WSI.to(device)
        label = label.to(device)

        with torch.no_grad():
            out = model(x_path=data_WSI)
            hazards = out["hazards"]
            survival = out["S"]
            Y_hat = out["Y_hat"]
            logits = out["logits"]
            h = out["h"]

        risk = (-torch.sum(survival, dim=1).cpu().numpy()).item()
        event_time_value = float(event_time.item())
        c_value = int(c.item())
        label_value = int(label.item())

        # case_id 可能是 tuple/list/string
        if isinstance(case_id, (list, tuple)):
            case_id_value = case_id[0]
        else:
            case_id_value = case_id

        all_risk_scores.append(risk)
        all_censorships.append(c_value)
        all_event_times.append(event_time_value)

        patient_results[case_id_value] = {
            "case_id": np.array(case_id_value),
            "risk": risk,
            "disc_label": label_value,
            "survival": event_time_value,
            "censorship": c_value,
            "slide_embedding": h.detach().cpu(),
        }

    all_risk_scores = np.asarray(all_risk_scores)
    all_censorships = np.asarray(all_censorships)
    all_event_times = np.asarray(all_event_times)

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    return patient_results, c_index

