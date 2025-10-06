import argparse
import os
import sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Get project root directory and add to Python module search path
_current_script_path = os.path.abspath(__file__)
_project_root = os.path.abspath(os.path.join(os.path.dirname(_current_script_path), '../../'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
print('Current working directory：', os.getcwd())
import json
import numpy as np
from src.models_dual_inter_traj.utils import visuaulize,seed_set,get_dct_matrix,gen_velocity,predict,getRandomPermuteOrder,getRandomRotatePoseTransform,getRandomScaleTransform
from lr import update_lr_multistep
from src.baseline_h36m_15to15.config import config
from src.models_dual_inter_traj.model import siMLPe as Model
from src.baseline_h36m_15to15.lib.datasets.dataset_mocap import DATA
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import  ensure_dir
import torch
from torch.utils.tensorboard import SummaryWriter
from src.baseline_h36m_15to15.test import mpjpe_test_regress,regress_pred,mpjpe_vim_test
import shutil

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default="+(2)", help='=exp name')
parser.add_argument('--dataset', type=str, default="others", help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', type=bool,default=True, help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--normalization',type=bool,default=False, help='=use only spatial fc')
parser.add_argument('--norm_way',type=str,default='first', help='=use only spatial fc')
parser.add_argument('--permute_p', type=bool, default=True, help='Permute P dimension')
parser.add_argument('--random_rotate', type=bool, default=True, help='Random rotation around world center')
parser.add_argument('--scale', type=bool, default=False, help='Scale human body data')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--interaction_interval', type=int, default=16, help='Interval between local and Global interactions, must be divisible by num')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--n_p', type=int, default=3)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--vis_every', type=int, default=25000000000000)
parser.add_argument('--save_every', type=int, default=250)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--use_distance', type=bool, default=True)
args = parser.parse_args()

# Create folder
expr_dir = os.path.join('exprs', args.exp_name)
if os.path.exists(expr_dir):
    shutil.rmtree(expr_dir)
os.makedirs(expr_dir, exist_ok=True)

# Ensure reproducibility
seed_set(args.seed)
torch.use_deterministic_algorithms(True)



# Configuration
config.norm_way=args.norm_way
config.normalization=args.normalization
config.batch_size = args.batch_size
config.dataset = args.dataset
config.n_p = args.n_p
config.vis_every = args.vis_every
config.save_every = args.save_every
config.print_every = args.print_every
config.debug = args.debug
config.device = args.device
config.expr_dir=expr_dir
config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num
config.motion_mlp.n_p = args.n_p
config.motion_mlp.use_distance = args.use_distance
config.motion_mlp.interaction_interval = args.interaction_interval
config.snapshot_dir=os.path.join(expr_dir, 'snapshot')
ensure_dir(config.snapshot_dir)# Create folder
config.vis_dir=os.path.join(expr_dir, 'vis')
ensure_dir(config.vis_dir)# Create folder
config.log_file=os.path.join(expr_dir, 'log.txt')
config.model_pth=args.model_path

#
writer = SummaryWriter()

# Get DCT matrix
dct_m,idct_m = get_dct_matrix(config.dct_len)
dct_m = torch.tensor(dct_m).float().to(config.device).unsqueeze(0)
idct_m = torch.tensor(idct_m).float().to(config.device).unsqueeze(0)
config.dct_m=dct_m
config.idct_m=idct_m

def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :
    if args.random_rotate:
        h36m_motion_input,h36m_motion_target=getRandomRotatePoseTransform(h36m_motion_input,h36m_motion_target)
    if args.scale:
        h36m_motion_input,h36m_motion_target=getRandomScaleTransform(h36m_motion_input,h36m_motion_target,0.5,1.5)
    if args.permute_p:
        h36m_motion_input,h36m_motion_target=getRandomPermuteOrder(h36m_motion_input,h36m_motion_target)
    motion_pred=predict(model,h36m_motion_input,config)#b,p,n,c
    b,p,n,c = h36m_motion_target.shape
    # Predicted pose
    motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3).reshape(-1,3)
    #GT
    h36m_motion_target = h36m_motion_target.to(config.device).reshape(b,p,n,config.n_joint,3).reshape(-1,3)
    # Calculate loss
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
        dmotion_pred = gen_velocity(motion_pred)# Calculate velocity
        
        motion_gt = h36m_motion_target.reshape(b,p,n,config.n_joint,3)
        dmotion_gt = gen_velocity(motion_gt)# Calculate velocity
        
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

# Create model
model = Model(config).to(device=config.device)
model.train()
print(">>> total params: {:.2f}M".format(
    sum(p.numel() for p in list(model.parameters())) / 1000000.0))


if config.dataset=="h36m":# Unused, dataset is not h36m
    pass
else:
    dataset = DATA( 'train', config.t_his,config.t_pred,n_p=config.n_p)
    eval_dataset_mocap = DATA( 'eval_mocap', config.t_his,config.t_pred_eval,n_p=config.n_p)
    eval_dataset_mupots=DATA('eval_mutpots',config.t_his,config.t_pred_eval,n_p=config.n_p)
    
# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)
# Create logger
def default_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return None  # or return a mark, or convert it to a serializable type
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

logger = get_logger(config.log_file, 'train')
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True,default=default_serializer))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ Training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

def write(metric_name, metric_val, iter,expr_dir):
    acc_log_dir = os.path.join(expr_dir, 'acc_log.txt')
    
    # Add seed information when writing for the first time
    if not os.path.exists(acc_log_dir):
        with open(acc_log_dir, 'w') as f:
            f.write(f'Seed: {args.seed}\n')
    
    # Append mode to write metric data
    with open(acc_log_dir, 'a') as llog:
        llog.write(f' {iter + 1}\n')
        line = f'{metric_name}: {np.mean(metric_val)} ' 
        line += ' '.join([f'{x}' for x in metric_val]) + '\n'
        llog.write(line)
        
while (nb_iter + 1) < config.cos_lr_total_iters:
    print(f"{nb_iter + 1} / {config.cos_lr_total_iters}")
    
    if config.dataset == 'h36m':
        pass
    else:
        train_generator = dataset.sampling_generator(num_samples=config.num_train_samples, batch_size=config.batch_size)
        data_source = train_generator

    for (h36m_motion_input, h36m_motion_target) in data_source:
        # B,P,T,JK
        h36m_motion_input=torch.tensor(h36m_motion_input).float()
        h36m_motion_target=torch.tensor(h36m_motion_target).float()
        
        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        
        if nb_iter == 9:
            print("第十个iter的loss:",loss)#0.22839394211769104
            
        avg_loss += loss
        avg_lr += current_lr
        # Print loss
        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0
        # Save model and evaluate model
        if (nb_iter + 1) % config.save_every ==  0 or nb_iter==0: 
            with torch.no_grad():
                # torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
                model.eval()
                if config.dataset=="h36m":
                    pass
                else:
                    print("begin test")
                    
                    eval_generator_mocap = eval_dataset_mocap.iter_generator(batch_size=config.batch_size)#按顺序遍历一次数据集
                    mpjpe_res_mocap,vim_res_mocap,jpe_res_mocap,ape_res_mocap,fde_res_mocap=mpjpe_vim_test(config, model, eval_generator_mocap,is_mocap=True,select_vim_frames=[1, 3, 7, 9, 14],select_mpjpe_frames=[3,9,15])#得到评估结果
                    
                    # if mpjpe_res_mocap[0]<min_mpjpe_mocap:#保存最好的模型
                    #     min_mpjpe_mocap=mpjpe_res_mocap[0]
                    #     torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
                    #     print("save model")
                    eval_generator_mupots = eval_dataset_mupots.iter_generator(batch_size=config.batch_size)
                    mpjpe_res_mupots,vim_res_mupots,jpe_res_mupots,ape_res_mupots,fde_res_mupots=mpjpe_vim_test(config, model, eval_generator_mupots,is_mocap=False,select_vim_frames=[1, 3, 7, 9, 14],select_mpjpe_frames=[3,9,15])
                    
                    write('mpjpe_mocap',mpjpe_res_mocap,nb_iter,config.expr_dir)
                    write('vim_mocap',vim_res_mocap,nb_iter,config.expr_dir)
                    write('jpe_mocap',jpe_res_mocap,nb_iter,config.expr_dir)
                    write('ape_mocap',ape_res_mocap,nb_iter,config.expr_dir)
                    write('fde_mocap',fde_res_mocap,nb_iter,config.expr_dir)
                    
                    write('mpjpe_mupots',mpjpe_res_mupots,nb_iter,config.expr_dir)
                    write('vim_mupots',vim_res_mupots,nb_iter,config.expr_dir)
                    write('jpe_mupots',jpe_res_mupots,nb_iter,config.expr_dir)
                    write('ape_mupots',ape_res_mupots,nb_iter,config.expr_dir)
                    write('fde_mupots',fde_res_mupots,nb_iter,config.expr_dir)
                    
                model.train()
        # Visualize model
        if ((nb_iter + 1) % config.vis_every ==  0 or nb_iter==0) and config.dataset!="h36m":
            model.eval()
            with torch.no_grad():  
                h36m_motion_input = eval_dataset_mocap.sample()[:,:,:config.t_his]
                h36m_motion_input=torch.tensor(h36m_motion_input,device=config.device).float()
                h36m_motion_input=h36m_motion_input[:1]#1，p,t,jk
                motion_pred=regress_pred(model,h36m_motion_input,config)
                
                b,p,n,c = motion_pred.shape
                motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
                h36m_motion_input=h36m_motion_input.reshape(b,p,config.t_his,config.n_joint,3)
                motion=torch.cat([h36m_motion_input,motion_pred],dim=2).cpu().detach().numpy()
                visuaulize(motion,f"iter:{nb_iter}",config.vis_dir)
                
                model.train()
        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
