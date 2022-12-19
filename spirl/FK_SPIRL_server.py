from collections import OrderedDict
import os
import matplotlib; matplotlib.use('Agg')
import torch
from shutil import copy
import numpy as np
import Custome_train
from spirl.components.params import get_args
from spirl.utils.general_utils import AttrDict

if __name__ == "__main__":

    config = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/hierarchical_cl",
        prefix = "minner",
        new_dir = False,
        dont_save = False,
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
    init_model =  Custome_train.ModelTrainer(args=config,cid=0)
    init_path = "/workspace/skill/experiments/skill_prior_learning/kitchen/7_client_1"
    for idx in range(100):
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
