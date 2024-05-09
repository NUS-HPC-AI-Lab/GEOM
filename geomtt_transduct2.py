import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import get_eval_pool
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.reparam_module import ReparamModule
import logging
import os
import random
import copy
import scipy
from gntk_cond import GNTK
from utils_graphsaint import DataGraphSAINT
from utils import *
import wandb 


class MetaGtt:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        n = int(data.feat_train.shape[0] * args.reduction_rate)

        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))

        if args.optimizer_con == 'Adam':
            self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        elif args.optimizer_con == 'SGD':
            self.optimizer_feat = torch.optim.SGD([self.feat_syn], lr=args.lr_feat, momentum=0.9)

        logging.info('adj_syn: {}, feat_syn: {}'.format((n, n), self.feat_syn.shape))

    def beta_mapping(self, epoch, upper_bound, lower_bound, end_epoch):

        if epoch >= end_epoch:
            return lower_bound

        x = epoch / end_epoch
        mapped_value = lower_bound + (upper_bound - lower_bound) / (1 + np.exp(-10 * (x - 0.5)))

        return mapped_value

    def expert_load(self):
        args = self.args
        expert_dir = args.buffer_path
        logging.info("Expert Dir: {}".format(expert_dir))

        if args.load_all:
            buffer = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))

        else:
            expert_files = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            file_idx = 0
            expert_idx = 0
            random.shuffle(expert_files)
            if args.max_files is not None:
                expert_files = expert_files[:args.max_files]  
            print("loading file {}".format(expert_files[file_idx]))
            buffer = torch.load(expert_files[file_idx])
            if args.max_experts is not None:
                buffer = buffer[:args.max_experts]
            random.shuffle(buffer)
            self.buffer = buffer

        return file_idx, expert_idx, expert_files  

    def synset_save(self):
        args = self.args
        
        with torch.no_grad():
            feat_save = self.feat_syn
            eval_labs = self.labels_syn

        feat_syn_eval, label_syn_eval = copy.deepcopy(feat_save.detach()), copy.deepcopy(
            eval_labs.detach())  # avoid any unaware modification

        adj_syn_eval = torch.eye(feat_syn_eval.shape[0]).to(self.device)

        return feat_syn_eval, adj_syn_eval, label_syn_eval
    
    def eval_synset(self, args):
    
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        device = torch.device(args.device)
        #logging.info('start!')
        if args.dataset in ['cora', 'citeseer']:
            args.epsilon = 0.05
        else:
            args.epsilon = 0.01
    
        data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
        if args.dataset in data_graphsaint:
            data = DataGraphSAINT(args.dataset)
            data_full = data.data_full
        else:
            data_full = get_dataset(args.dataset)
            data = Transd2Ind(data_full)
    
        res_val = []
        res_test = []
        nlayer = 2
        for i in range(args.nruns):
            best_acc_val, best_acc_test = self.test(args, data, device, model_type=args.test_model_type, nruns=i)
            # _,_ = self.test_lr(args, data, device, model_type=args.test_model_type, nruns=i)


            res_val.append(best_acc_val)
            res_test.append(best_acc_test)
        res_val = np.array(res_val)
        res_test = np.array(res_test)
        logging.info('Model:{}, Layer: {}'.format(args.test_model_type, nlayer))
        logging.info('TEST: Full Graph Mean Accuracy: {:.6f}, STD: {:.6f}'.format(res_test.mean(), res_test.std()))
        logging.info('TEST: Valid Graph Mean Accuracy: {:.6f}, STD: {:.6f}'.format(res_val.mean(), res_val.std()))
    
        return res_val, res_test
    
    def SoftCrossEntropy(self, inputs, target, reduction='average'):
        input_log_likelihood = - inputs
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss


    def test(self, args, data, device, model_type, nruns):
    
        if args.whole_data != 1:
            feat_syn, adj_syn, labels_syn = self.synset_save()
            adj_syn = torch.eye(feat_syn.shape[0]).to(device)
            if type(adj_syn) is not torch.Tensor:
                feat_syn, adj_syn, labels_syn = utils.to_tensor(feat_syn, adj_syn, labels_syn, device=device)
            else:
                feat_syn, adj_syn, labels_syn = feat_syn.to(device), adj_syn.to(device), labels_syn.to(device)
            if model_type == 'MLP':
                adj_syn = adj_syn - adj_syn
                model_class = GCN
            else:
                model_class = eval(model_type)
    
            if utils.is_sparse_tensor(adj_syn):
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=True)
            else:
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn)
            adj_syn = adj_syn_norm
            weight_decay = args.test_wd
            lr = args.test_lr_model
    
        else:
            logging.info('THIS IS THE ORIGINAL WHOLE DATA...')
            features, adj, labels = data.feat_full, data.adj_full, data.labels_full
            features, adj, labels = utils.to_tensor(features, adj, labels, device=device)
            feat_syn, labels_syn = features, labels
            if model_type == 'MLP':
                adj = adj - adj
                model_class = GCN
            else:
                model_class = eval(model_type)
            if utils.is_sparse_tensor(adj):
                adj_syn_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_syn_norm = utils.normalize_adj_tensor(adj)
    
            adj_syn = adj_syn_norm
            weight_decay = 5e-4
            lr = 0.01
    
    
        #dropout = 0.5 if args.dataset in ['reddit'] else args.test_dropout
        dropout = args.test_dropout
    
        model = model_class(nfeat=feat_syn.shape[1], nhid=args.test_hidden, dropout=dropout, nlayers=args.test_nlayers,
                            nclass=data.nclass, device=device).to(device)
    
    
        if args.test_opt_type=='Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif args.test_opt_type=='SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
    
        best_acc_val = best_acc_test = best_acc_it = 0
    
        train_iters = args.test_model_iters
    
        for i in range(train_iters):
            if i == train_iters // 2 and args.lr_decay == 1:
                lr = args.test_lr_model * 0.5
                if args.test_opt_type == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                elif args.test_opt_type == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
            if args.whole_data == 1:
                model.train()
                optimizer.zero_grad()
                _,output_syn = model.forward(feat_syn, adj_syn)
                loss_train = F.nll_loss(output_syn[data.idx_train], labels_syn[data.idx_train])
                acc_syn = utils.accuracy(output_syn[data.idx_train], labels_syn[data.idx_train])
            else:
                model.train()
                optimizer.zero_grad()
                _,output_syn = model.forward(feat_syn, adj_syn)
                # loss_train = F.nll_loss(output_syn, labels_syn) torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
                # tag
                # loss_train = self.SoftCrossEntropy(output_syn, labels_syn)
                if args.soft_label == True:

                    loss_train = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(output_syn, labels_syn)
                    acc_syn = utils.accuracy(output_syn, torch.argmax(labels_syn, dim=1))
                else:
                    loss_train = F.nll_loss(output_syn, labels_syn)
                    acc_syn = utils.accuracy(output_syn, labels_syn)

                # acc_syn = utils.accuracy(output_syn, torch.argmax(labels_syn, dim=1))
    
            loss_train.backward()
            optimizer.step()
    
            if i % 20 == 0:
    
                # logging.info('Epoch {}, training loss: {}, training acc: {}'.format(i, loss_train.item(), acc_syn.item()))
                model.eval()
                labels_test = torch.LongTensor(data.labels_test).to(device)
                labels_val = torch.LongTensor(data.labels_val).to(device)
    
                if args.dataset in ['reddit', 'flickr']:
                    _,output_val = model.predict(data.feat_val, data.adj_val)
                    acc_val = utils.accuracy(output_val, labels_val)
    
                    _, output_test = model.predict(data.feat_test, data.adj_test)
                    acc_test = utils.accuracy(output_test, labels_test)
    
                    if acc_val.item() > best_acc_val:

                        best_acc_val = acc_val.item()
                        best_acc_test = acc_test.item()
                        best_acc_it = i
                else:
                    # Full graph
                    _,output = model.predict(data.feat_full, data.adj_full)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                    acc_test = utils.accuracy(output[data.idx_test], labels_test)
    
                    if acc_val.item() > best_acc_val:

                        best_acc_val = acc_val.item()
                        best_acc_test = acc_test.item()
                        best_acc_it = i
    
        logging.info('FINAL BEST ACC TEST: {:.6f} with in {}-iteration'.format(best_acc_test,best_acc_it))
        return best_acc_val, best_acc_test

    def distill(self, writer):

        args = self.args
        data = self.data

        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        feat_init, adj_init, labels_init = self.get_coreset_init(features, adj, labels)
        feat_init, adj_init, labels_init = utils.to_tensor(feat_init, adj_init, labels_init, device=self.device)
        features_tensor, adj_tensor, labels_tensor = utils.to_tensor(features, adj, labels, device=self.device)
        adj_tensor_norm = utils.normalize_adj_tensor(adj_tensor, sparse=True)

        self.feat_syn.data.copy_(feat_init)
        self.labels_syn = labels_init
        self.adj_syn_init = adj_init

        file_idx, expert_idx, expert_files = self.expert_load()  
        if args.soft_label == True:
            # -------------------------------------softlabel---------------------------------start---------------------------------------------------------------------------------#
            if args.dataset in ['ogbn-arxiv']:
                    model_4_soft = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                                device=self.device).to(self.device)
            else:
                    if args.condense_model == 'SGC':
                        model_4_soft = SGC(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                    nclass=data.nclass, dropout=args.student_dropout,
                                    nlayers=args.student_nlayers, with_bn=False,
                                    device=self.device).to(self.device)
                    elif args.condense_model == 'GCN':
                        model_4_soft = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                    nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                                    device=self.device).to(self.device)
            
            
            model_4_soft = ReparamModule(model_4_soft)    

            model_4_soft.eval()
            Temp_params = self.buffer[0][-1]
            Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
            # if args.distributed:
            #     Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
            

            # adj_syn_norm = utils.normalize_adj_tensor(self.adj_syn_init, sparse=True)
            # adj_syn_input = adj_syn_norm 
            adj_syn = torch.eye(self.feat_syn.shape[0]).to(self.device)
            adj_syn_cal_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
            adj_syn_input = adj_syn_cal_norm


            feat_4_soft, adj_4_soft = copy.deepcopy(self.feat_syn.detach()), copy.deepcopy(
                adj_syn_input.detach())
            feat_out, label_soft = model_4_soft.forward(feat_4_soft, adj_4_soft, flat_param=Initialize_Labels_params)
            
            max_pred, pred_lab = torch.max(label_soft, dim=1)
    
            for i in range(labels_init.shape[0]):
                if pred_lab[i] != labels_init[i]:
                    label_soft[i][labels_init[i]] = max_pred[i]
                    # label_soft[i].fill_(0)
                    # label_soft[i][labels_init[i]] = 1
            
            self.labels_syn = copy.deepcopy(label_soft.detach()).to(args.device).requires_grad_(True)
            self.labels_syn.requires_grad=True
            self.labels_syn = self.labels_syn.to(args.device)
            
            acc = np.sum(np.equal(np.argmax(label_soft.cpu().data.numpy(), axis=-1), labels_init.cpu().data.numpy()))
            print('InitialAcc:{}'.format(acc/len(self.labels_syn)))
            
            self.optimizer_label = torch.optim.SGD([self.labels_syn], lr=args.lr_y, momentum=0.9)
#-------------------------------------softlabel-------------------------------------------------------end-----------------------------------------------------------------#
        else:
            self.labels_syn = labels_init

        # print("max:",torch.max(self.labels_syn),"min",torch.min(self.labels_syn))
        self.syn_lr = torch.tensor(args.lr_student).to(self.device)

        if args.optim_lr == 1:
            self.syn_lr = self.syn_lr.detach().to(self.device).requires_grad_(True)
            
            if args.optimizer_lr == 'Adam':
                optimizer_lr = torch.optim.Adam([self.syn_lr], lr=args.lr_lr)
            elif args.optimizer_lr == 'SGD':
                optimizer_lr = torch.optim.SGD([self.syn_lr], lr=args.lr_lr, momentum=0.5)

        eval_it_pool = np.arange(0, args.ITER + 1, args.eval_interval).tolist()
        model_eval_pool = get_eval_pool(args.eval_type, args.condense_model, args.eval_model)
        accs_all_exps = dict()  # record performances of all experiments
        for key in model_eval_pool:
            accs_all_exps[key] = []

        best_accs_test = {m: 0 for m in model_eval_pool}
        best_accs_test_iter = {m: 0 for m in model_eval_pool}
        best_model_std_test = {m: 0 for m in model_eval_pool}

        best_loss = 1.0
        best_loss_it = 0
        adj_syn_norm_key = {'0': 0}

        for it in range(0, args.ITER + 1):
            #logging.info(adj_syn_norm_key['0'])
            if args.dataset in ['ogbn-arxiv']:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                            nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                            device=self.device).to(self.device)
                model_4_clom = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                            nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                            device=self.device).to(self.device)
            else:
                if args.condense_model == 'SGC':
                    model = SGC(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                nclass=data.nclass, dropout=args.student_dropout,
                                nlayers=args.student_nlayers, with_bn=False,
                                device=self.device).to(self.device)
                    model_4_clom = SGC(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                nclass=data.nclass, dropout=args.student_dropout,
                                nlayers=args.student_nlayers, with_bn=False,
                                device=self.device).to(self.device)
                elif args.condense_model == 'GCN':
                    model = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                                device=self.device).to(self.device)
                    model_4_clom = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                                device=self.device).to(self.device)
            # model.initialize()

            model = ReparamModule(model)
            model_4_clom = ReparamModule(model_4_clom)

            model.train()

            num_params = sum([np.prod(p.size()) for p in (model.parameters())])

            if args.load_all:
                expert_trajectory = self.buffer[np.random.randint(0, len(self.buffer))]
            else:
                expert_trajectory = self.buffer[expert_idx]  
                expert_idx += 1
                if expert_idx == len(self.buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_files)
                    print("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1:
                        del self.buffer
                        self.buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        self.buffer = self.buffer[:args.max_experts]
                    random.shuffle(self.buffer)

                    
            if args.expanding_window:
                Upper_Bound = args.max_start_epoch_s + it
                Upper_Bound = min(Upper_Bound, args.max_start_epoch)                                       
            else:
                Upper_Bound = args.max_start_epoch
            
            print(Upper_Bound)
            
            np.random.seed(it)
            start_epoch = np.random.randint(args.min_start_epoch, Upper_Bound) # 100-1600 


            np.random.seed(args.seed_student)
            
            start_epoch = start_epoch // 10
            starting_params = expert_trajectory[start_epoch]

            if args.interval_buffer == 1:
                print(start_epoch + args.expert_epochs // 10)
                target_params = expert_trajectory[start_epoch + args.expert_epochs // 10]
                print(start_epoch + args.expert_epochs // 10)
            else:
                target_params = expert_trajectory[start_epoch + args.expert_epochs]

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)


            
            target_params_4_clom = expert_trajectory[-1]
            
            target_params_4_clom = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params_4_clom], 0)
            

            params_dict = dict(model_4_clom.named_parameters())


            for (name, param) in params_dict.items():
                    param.data.copy_(target_params_4_clom)
                    

            model_4_clom.load_state_dict(params_dict)
            
            for param in model_4_clom.parameters():
                param.requires_grad = False

            student_params = [
                torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            param_loss_list = []
            param_dist_list = []
            logging.info('it:{}--feat_max = {:.4f}, feat_min = {:.4f}'.format(it, torch.max(self.feat_syn),
                                                                              torch.min(self.feat_syn)))

            # if it == 0 and args.dataset in ['cora']:
            #     feat_syn = self.feat_syn
            #     adj_syn_norm = utils.normalize_adj_tensor(self.adj_syn_init, sparse=True)
            #     adj_syn_input = adj_syn_norm

            # else:
            feat_syn = self.feat_syn
            adj_syn = torch.eye(feat_syn.shape[0]).to(self.device)
            adj_syn_cal_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
            adj_syn_input = adj_syn_cal_norm

            # tag
            for step in range(args.syn_steps):
                forward_params = student_params[-1]
                feat_out, output_syn = model.forward(feat_syn, adj_syn_input, flat_param=forward_params)

                # print("输出:",output_syn.shape)
                # # print("输出:",output_syn[0])

                # loss_syn = F.nll_loss(output_syn, self.labels_syn)
                
                if args.soft_label == True:
                    loss_syn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(output_syn, self.labels_syn)
                    acc_syn = utils.accuracy(output_syn, torch.argmax(self.labels_syn, dim=1))
                else:
                    loss_syn = F.nll_loss(output_syn, self.labels_syn)
                    acc_syn = utils.accuracy(output_syn, self.labels_syn)


                grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]


                # student_params.append(student_params[-1] - self.syn_lr * grad)
                student_params[-1] = student_params[-1] - self.syn_lr * grad
                if step % 500 == 0:
                    _, output_test = model.forward(features_tensor, adj_tensor_norm, flat_param=student_params[-1])
                    acc_test = utils.accuracy(output_test[data.idx_test], labels_tensor[[data.idx_test]])
                    logging.info('loss = {:.4f},acc_syn = {:.4f},acc_test = {:.4f}'.format(loss_syn.item(),
                                                                                           acc_syn.item(),
                                                                                           acc_test.item()))

            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += torch.norm(student_params[-1] - target_params, 2)
            param_dist += torch.norm(starting_params - target_params, 2)

            # param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")  
            
            # param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")


            
            
            # param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
            
            
            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss
            # total_loss = grand_loss + ntk_loss
            feat_out_clom, output_clom = model_4_clom.forward(feat_syn, adj_syn_input, flat_param=target_params_4_clom)
            # loss_clom = F.nll_loss(output_clom, self.labels_syn)
            
            if args.soft_label == True:
                loss_clom = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(output_clom, self.labels_syn)
            else:
                loss_clom = F.nll_loss(output_clom, self.labels_syn)


            # total_loss = grand_loss + ntk_loss
            # total_loss = grand_loss
            # beta = self.beta_mapping(it, args.high,args.low,args.end)
            total_loss = grand_loss + args.beta*loss_clom


            self.optimizer_feat.zero_grad()
            if args.soft_label == True:
                self.optimizer_label.zero_grad()

            if args.optim_lr == 1:
                optimizer_lr.zero_grad()

            # grand_loss.backward()
            total_loss.backward()
            # print(torch.min(self.feat_syn), torch.max(self.feat_syn), torch.min(self.feat_syn.grad), torch.max(self.feat_syn.grad))
            self.optimizer_feat.step()

            if args.soft_label == True:
                self.optimizer_label.step()




            logging.info('torch.sum(self.feat_syn) = {}'.format(torch.sum(self.feat_syn)))
            if args.optim_lr == 1:
                optimizer_lr.step()
                writer.add_scalar('student_lr_change', self.syn_lr.item(), it)
            if torch.isnan(total_loss) or torch.isnan(grand_loss):
                break  # Break out of the loop if either is NaN
            if it % 1 == 0:
                logging.info(
                    "Iteration {}: Total_Loss = {:.4f}, Grand_Loss={:.4f}, Start_Epoch= {}, Student_LR = {:6f}".format(
                        it,
                        total_loss.item(),
                        grand_loss.item(),
                        start_epoch,
                        self.syn_lr.item()))


            if it in eval_it_pool and it > 0:
                for model_eval in model_eval_pool:
                    logging.info(
                        'Evaluation: model_train = {}, model_eval = {}, iteration = {}'.format(args.condense_model,
                                                                                               model_eval,
                                                                                               it))
                    
                    best_acc_eval, best_acc_test = self.eval_synset(args)
                    feat_syn_save, adj_syn_save, label_syn_save = self.synset_save()
                    
                    

                    best_acc_eval = np.array(best_acc_eval)
                    best_acc_test = np.array(best_acc_test)

                    best_acc_eval = np.mean(best_acc_eval)
                    best_acc_test = np.mean(best_acc_test)
                    acc_eval_std = np.std(best_acc_eval)
                    # writer.add_scalar('ntk_acc_eval_curve', best_acc_eval, it)


                    if best_acc_eval > best_accs_eval[model_eval]:
                        best_accs_test[model_eval] = best_acc_test
                        best_accs_test_iter[model_eval] = it
                        torch.save(adj_syn_save,
                                    f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')
                        torch.save(feat_syn_save,
                                    f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')
                        torch.save(label_syn_save,
                                    f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')
                        
                        logging.info('new best test_acc occurs, the eval_acc is {:.4f}, the test_acc is {:.4f}, the iter is {:.2f} ! ! !'.format(best_acc_eval * 100.0, best_acc_test * 100.0, it))

                     
          
                    # if best_acc_eval > best_accs_test[model_eval]:
                    #     best_accs_test[model_eval] = best_acc_eval
                    #     best_accs_test_iter[model_eval] = it
                    #     torch.save(adj_syn_save,
                    #                f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')
                    #     torch.save(feat_syn_save,
                    #                f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')
                    #     torch.save(label_syn_save,
                    #                f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_best_eval_acc_{args.seed_student}_ours.pt')
                        
                    #     logging.info('new best eval_acc occurs, the eval_acc is {:.4f}, the test_acc is {:.4f}, the iter is {:.2f} ! ! !'.format(best_acc_eval * 100.0, best_acc_test * 100.0, it))


            if it % 1000 == 0 or it==args.ITER:
                feat_syn_save, adj_syn_save, label_syn_save = self.synset_save()
                torch.save(adj_syn_save,
                           f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}_ours.pt')
                torch.save(feat_syn_save,
                           f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}_ours.pt')
                torch.save(label_syn_save,
                           f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}_ours.pt')
            # for _ in student_params:
            #     del _

            if grand_loss.item() < best_loss:
                best_loss = grand_loss.item()
                best_loss_it = it

            writer.add_scalar('grand_loss_curve', grand_loss.item(), it)
            # torch.cuda.empty_cache()

            # gc.collect()


        for model_eval in model_eval_pool:
            logging.info('Evaluation ACC: {} for best eval_acc = {:.5f}, within {} iter'.format(
                model_eval, best_accs_test[model_eval], best_accs_test_iter[model_eval]))

        logging.info('This is the smallest loss: {:.06f} within {} iteration'.format(best_loss, best_loss_it))

    def get_coreset_init(self, features, adj, labels):
        logging.info('Loading from: {}'.format(self.args.coreset_init_path))
        idx_selected_train = np.load(
            f'{self.args.coreset_init_path}/idx_{self.args.dataset}_{self.args.reduction_rate}_{self.args.coreset_method}_{self.args.coreset_seed}.npy')
        feat_train = features[idx_selected_train]
        adj_train = adj[np.ix_(idx_selected_train, idx_selected_train)]
        labels_train = labels[idx_selected_train]
        return feat_train, adj_train, labels_train


def one_hot(x,
            num_classes,
            center=True,
            dtype=np.float32):
    assert len(x.shape) == 1
    one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
    if center:
        one_hot_vectors = one_hot_vectors - 1. / num_classes
    return one_hot_vectors


def calc(gntk, feat1, feat2, diag1, diag2, A1, A2):
    return gntk.gntk(feat1, feat2, diag1, diag2, A1, A2)


def loss_acc_fn_train(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    # print(k_ss.device, torch.abs(torch.tensor(reg)).to(k_ss.device),torch.trace(k_ss).device, torch.eye(k_ss.shape[0]).device)
    k_ss_reg = (k_ss + torch.abs(torch.tensor(reg)).to(k_ss.device) * torch.trace(k_ss).to(k_ss.device) * torch.eye(
        k_ss.shape[0]).to(k_ss.device) / k_ss.shape[0])
    pred = torch.matmul(k_ts[data.idx_train, :].cuda(), torch.matmul(torch.linalg.inv(k_ss_reg).cuda(),
                                                                     torch.from_numpy(y_support).to(
                                                                         torch.float64).cuda()))
    mse_loss = torch.nn.functional.mse_loss(pred.to(torch.float64).cuda(),
                                            torch.from_numpy(y_target).to(torch.float64).cuda(), reduction="mean")
    acc = 0
    return mse_loss, acc


def loss_acc_fn_eval(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    k_ss_reg = (k_ss + np.abs(reg) * np.trace(k_ss) * np.eye(k_ss.shape[0]) / k_ss.shape[0])
    pred = np.dot(k_ts, np.linalg.inv(k_ss_reg).dot(y_support))
    mse_loss = 0.5 * np.mean((pred - y_target) ** 2)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_target, axis=1))
    return mse_loss, acc
