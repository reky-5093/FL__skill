from collections import OrderedDict
import os
import matplotlib; matplotlib.use('Agg')
import torch
from shutil import copy
import numpy as np
import pandas as pd
import Custome_train2
from spirl.components.params import get_args
from spirl.utils.general_utils import AttrDict
from spirl.components.checkpointer import load_by_key, freeze_modules
from torch import autograd
import time
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict

if __name__ == "__main__":

    config = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/hierarchical_cl",
        prefix = "minner4",
        new_dir = False,
        dont_save = True,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = -1,
        strict_weight_loading = True,
        deterministic = False,
        log_interval = 500,
        per_epoch_img_logs = 1,
        val_data_size = 160,
        val_interval = 5,
        detect_anomaly = False,
        feed_random_data = False,
        train_loop_pdb = False,
        debug = False,
        save2mp4 = False
    )

    init_path = "/home/workspace/skill/experiments/skill_prior_learning/kitchen/non-iid_3"
    
    init_model =  Custome_train2.ModelTrainer(args=config,cid=0)
    key_value = init_model.model.state_dict().keys()
    '''

    
    
    for idx in range(789):
        cnt = 10 * (idx + 1)
        load_dict = np.load(os.path.join(init_path,"weights","round-{0}-weights.npz".format(cnt)))
        torch.device('cuda')
        params_dict = zip(init_model.model.state_dict().keys(), load_dict)
        state_dict = OrderedDict({k: torch.Tensor(load_dict[v]) for k, v in params_dict})
        load_dict.close()
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del( state_dict[d] )
        init_model.model.load_state_dict(state_dict)
        #init_model.val()
        state = {
            'epoch': 99,
            'global_step': 0,
            'state_dict': init_model.model.state_dict(),
            'optimizer': init_model.optimizer.state_dict(),
        }
        save_path = os.path.join(init_path,str(cnt),"weights")
        os.makedirs(save_path,exist_ok=True)
        torch.save(state, save_path+"/weights_ep99.pth")
    '''

    init_model.model.load_state_dict(torch.load("/home/workspace/skill/experiments/skill_prior_learning/kitchen/hierarchical_cl/normal_4/weights/weights_ep99.pth")['state_dict'])
    init_path = "/home/workspace/skill/experiments/skill_prior_learning/kitchen/non-iid_3"
    load_dict = np.load(os.path.join(init_path,"weights","round-1000-weights.npz"))
    torch.device('cuda')
    normal_state_dict = init_model.model.state_dict()
    params_dict = zip(init_model.model.state_dict().keys(), load_dict)
    state_dict = OrderedDict({k: torch.Tensor(load_dict[v]) for k, v in params_dict})
    load_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d:
            l.append(d)
        if "decoder" in d :
            state_dict[d] = normal_state_dict[d]
    for d in l :
        del(state_dict[d])
    init_model.model.load_state_dict(state_dict,strict = True)
    #init_model.val()
    
    state = {
        'epoch': 99,
        'global_step': 0,
        'state_dict': init_model.model.state_dict(),
        'optimizer': init_model.optimizer.state_dict(),
    }
    save_path = os.path.join("/home/workspace/skill/experiments/skill_prior_learning/kitchen/N_decoder_F_prior","weights")
    os.makedirs(save_path,exist_ok=True)
    torch.save(state, save_path+"/weights_ep99.pth")
    '''
    
    init_model1 =  Custome_train2.ModelTrainer(args=config,cid=0)
    init_model2 =  Custome_train2.ModelTrainer(args=config,cid=0)
    init_model3 =  Custome_train2.ModelTrainer(args=config,cid=0)
    init_model1.model.load_state_dict(torch.load("/home/workspace/skill/experiments/skill_prior_learning/kitchen/hierarchical_cl/normal_4/weights/weights_ep99.pth")['state_dict'])
    init_model3.model.load_state_dict(torch.load("/home/workspace/skill/experiments/skill_prior_learning/kitchen/sample_decoder/weights/weights_ep99.pth")['state_dict'])
    init_path = "/home/workspace/skill/experiments/skill_prior_learning/kitchen/non-iid_3"
    load_dict = np.load(os.path.join(init_path,"weights","round-1000-weights.npz"))
    torch.device('cuda')
    params_dict = zip(init_model2.model.state_dict().keys(), load_dict)
    state_dict = OrderedDict({k: torch.Tensor(load_dict[v]) for k, v in params_dict})
    load_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    init_model2.model.load_state_dict(state_dict,strict = True)
    
    manuple_load_dict = init_model1.model.state_dict()
    
    l = []
    for d in manuple_load_dict :
        if ("num_batches_tracked" in d) or ("q.0" in d) or ("p.0" in d) or ("q.1" in d):
            l.append(d)
    for d in l :
        del(manuple_load_dict[d])
    init_model1.model.load_state_dict(manuple_load_dict,strict = False)
    #init_model.val()
   
    state = {
        'epoch': 99,
        'global_step': 0,
        'state_dict': init_model1.model.state_dict(),
        'optimizer': init_model1.optimizer.state_dict(),
    }
    save_path = os.path.join("/home/workspace/skill/experiments/skill_prior_learning/kitchen/FL_decoder","weights")
    os.makedirs(save_path,exist_ok=True)
    torch.save(state, save_path+"/weights_ep99.pth")
    
    dataset_class = init_model1.conf.data.dataset_spec.dataset_class
    phase = 'val'
    val_loader = dataset_class(init_model1.data_dir, init_model1.conf.data, resolution=init_model1.model.resolution,
                            phase=phase, shuffle=phase == "train", dataset_size=160). \
        get_data_loader(init_model1._hp.batch_size, 1)
    print('Running Testing')
    start = time.time()
    results_ls = list
    init_model1.model_test.load_state_dict(init_model1.model.state_dict())
    init_model2.model_test.load_state_dict(init_model2.model.state_dict())
    init_model3.model_test.load_state_dict(init_model3.model.state_dict())
    init_model1.model_test.eval()
    init_model2.model_test.eval()
    init_model3.model_test.eval()
    with autograd.no_grad():
        for sample_batched in val_loader:
                inputs = AttrDict(map_dict(lambda x: x.to(init_model2.device), sample_batched))
                # run non-val-mode model (inference) to check overfitting
                output1 = init_model1.model_test(inputs)
                output2 = init_model2.model_test(inputs)
                output3 = init_model3.model_test(inputs)
                break
    q_mu = []
    q_sigma = []
    q_hat_mu = []
    q_hat_sigma = []
    reconstruction = []
    for a,b in zip(np.array(output1.q.mu.cpu()),np.array(output2.q.mu.cpu())):
        dist = np.linalg.norm(a-b)
        q_mu.append(dist)
    for a,b in zip(np.array(output1.q.sigma.cpu()),np.array(output2.q.sigma.cpu())):
        dist = np.linalg.norm(a-b)
        q_sigma.append(dist)
    for a,b in zip(np.array(output1.q_hat.mu.cpu()),np.array(output2.q_hat.mu.cpu())):
        dist = np.linalg.norm(a-b)
        q_hat_mu.append(dist)
    for a,b in zip(np.array(output1.q_hat.sigma.cpu()),np.array(output2.q_hat.sigma.cpu())):
        dist = np.linalg.norm(a-b)
        q_hat_sigma.append(dist)
    for a,b in zip(np.array(output1.reconstruction.cpu()),np.array(output3.reconstruction.cpu())):
        dist = np.linalg.norm(a-b)
        reconstruction.append(dist)       
    print("Mean q_mu is {0}".format(np.mean(q_mu)))
    print("Var q_mu is {0}".format(np.var(q_mu)))
    print("Mean q_sigma is {0}".format(np.mean(q_sigma)))
    print("Var q_sigma is {0}".format(np.var(q_sigma)))
    print("Mean q_mu is {0}".format(np.mean(q_hat_mu)))
    print("Var q_mu is {0}".format(np.var(q_hat_mu)))
    print("Mean q_sigma is {0}".format(np.mean(q_hat_sigma)))
    print("Var q_sigma is {0}".format(np.var(q_hat_sigma)))
    print("Mean reconstruction is {0}".format(np.mean(reconstruction)))
    print("Var reconstruction is {0}".format(np.var(reconstruction)))
    student_card = pd.DataFrame({'q_mu': q_mu,
                             'q_sigma':q_sigma,
                             'q_hat_mu': q_hat_mu,
                             'q_hat_sigma':q_hat_sigma,
                             'reconstruction':reconstruction})
    student_card.to_csv("/home/workspace/skill/experiments/Euclidean distance.csv")
    '''
