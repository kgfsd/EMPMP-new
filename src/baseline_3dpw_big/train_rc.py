import argparse
import os
import sys
import json
import warnings
import shutil
import logging  # Import logging module
from easydict import EasyDict as edict

# Set environment variable for CUDA determinism
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Get project root directory and add to Python module search path
_current_script_path = os.path.abspath(__file__)
_project_root = os.path.abspath(os.path.join(os.path.dirname(_current_script_path), '../../'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
print('Current working directory：', os.getcwd())

# Import your modules
from src.models_dual_inter_traj_big.utils import Get_RC_Data, visuaulize, visuaulize_bianhao, seed_set, get_dct_matrix, \
    gen_velocity, predict, update_metric, getRandomPermuteOrder, getRandomRotatePoseTransform
from lr import update_lr_multistep
from src.baseline_3dpw_big.config import config
from src.models_dual_inter_traj_big.model import siMLPe as Model
from src.baseline_3dpw_big.lib.dataset.dataset_3dpw import get_3dpw_dataloader
from src.baseline_3dpw_big.lib.dataset.data_utils import path_to_repo
from src.baseline_3dpw_big.lib.utils.logger import get_logger, print_and_log_info
from src.baseline_3dpw_big.lib.utils.pyt_utils import ensure_dir
import torch
from torch.utils.tensorboard import SummaryWriter
from src.baseline_3dpw_big.test import vim_test, random_pred, mpjpe_vim_test

# 添加混合人数数据集支持
from test_mixed_people import MixedPeopleDataset, collate_mixed_batch

# Ignore all warnings
warnings.filterwarnings("ignore")


# --- Auxiliary functions (can be outside if __name__ == '__main__': block) ---

def close_all_file_handlers_in_dir(target_dir):
    """
    Closes all logging.FileHandlers if their log files are within target_dir,
    and removes them from their respective loggers. This helps prevent
    PermissionError on shutil.rmtree due to file locks.
    """
    abs_target_dir = os.path.abspath(target_dir)

    # Iterate over all loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        # Skip 'root' logger as it might have handlers we don't want to touch
        if logger_name == 'root':
            continue

        logger = logging.getLogger(logger_name)
        handlers_to_remove = []
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler_file_path = handler.baseFilename
                abs_handler_file_path = os.path.abspath(handler_file_path)

                # Check if the handler's file is within the target directory or is the directory itself
                if abs_handler_file_path.startswith(abs_target_dir + os.sep) or abs_handler_file_path == abs_target_dir:
                    print(f"DEBUG: Closing logging.FileHandler for: {handler_file_path}")
                    handler.close()
                    handlers_to_remove.append(handler)

        # Remove closed handlers from the logger
        for handler in handlers_to_remove:
            logger.removeHandler(handler)


def default_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write(metric_name, metric_val, iter, llog):
    llog.write(''.join(str(iter + 1) + '\n'))

    line = f'{metric_name}:'
    line += str(metric_val.mean()) + ' '
    for ii in metric_val:
        line += str(ii) + ' '
    line += '\n'
    llog.write(''.join(line))
    llog.flush()


def train_step(h36m_motion_input, h36m_motion_target, padding_mask, model, optimizer, nb_iter, total_iter, max_lr,
               min_lr, writer_obj):
    global latest_people_stats
    
    # 计算并显示当前批次的实际人数信息
    batch_size = padding_mask.shape[0]
    actual_people_counts = padding_mask.sum(dim=1).cpu().numpy()  # 每个样本的实际人数
    max_people_in_batch = actual_people_counts.max()
    min_people_in_batch = actual_people_counts.min()
    avg_people_in_batch = actual_people_counts.mean()
    
    # 更新全局统计信息
    latest_people_stats['min'] = min_people_in_batch
    latest_people_stats['max'] = max_people_in_batch
    latest_people_stats['avg'] = avg_people_in_batch
    latest_people_stats['counts'] = actual_people_counts.tolist()
    
    # 每100个iteration输出一次人数信息
    if nb_iter % 100 == 0:
        print(f"Batch people info - Min: {min_people_in_batch}, Max: {max_people_in_batch}, Avg: {avg_people_in_batch:.1f}")
    
    if args.random_rotate:
        h36m_motion_input, h36m_motion_target = getRandomRotatePoseTransform(h36m_motion_input, h36m_motion_target)
    if args.permute_p:
        h36m_motion_input, h36m_motion_target = getRandomPermuteOrder(h36m_motion_input, h36m_motion_target)
    if config.rc:
        h36m_motion_input, h36m_motion_target = Get_RC_Data(h36m_motion_input, h36m_motion_target)

    motion_pred = predict(model, h36m_motion_input, config, h36m_motion_target, padding_mask)  # b,p,n,c
    b, p, n, c = h36m_motion_target.shape

    # Predicted pose
    motion_pred = motion_pred.reshape(b, p, n, config.n_joint, 3).reshape(-1, 3)
    # GT
    h36m_motion_target = h36m_motion_target.to(config.device).reshape(b, p, n, config.n_joint, 3).reshape(-1, 3)
    # mask:b,p -> expanded to cover all joints and frames
    expanded_mask = padding_mask.unsqueeze(-1).repeat(1, 1, n * config.n_joint).reshape(-1)
    
    # Calculate loss only for valid (non-padded) people
    valid_errors = torch.norm(motion_pred - h36m_motion_target, dim=1)
    if expanded_mask.sum() > 0:  # Ensure there are valid people
        loss = torch.mean(valid_errors[expanded_mask])
    else:
        loss = torch.tensor(0.0, device=motion_pred.device, requires_grad=True)

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b, p, n, config.n_joint, 3)
        dmotion_pred = gen_velocity(motion_pred)  # Calculate velocity

        motion_gt = h36m_motion_target.reshape(b, p, n, config.n_joint, 3)
        dmotion_gt = gen_velocity(motion_gt)  # Calculate velocity

        # Mask for velocity (one less frame in time dimension)
        expanded_mask_vel = padding_mask.unsqueeze(-1).repeat(1, 1, (n - 1) * config.n_joint).reshape(-1)
        valid_vel_errors = torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), dim=1)
        if expanded_mask_vel.sum() > 0:  # Ensure there are valid velocity samples
            dloss = torch.mean(valid_vel_errors[expanded_mask_vel])
        else:
            dloss = torch.tensor(0.0, device=motion_pred.device, requires_grad=True)
        loss = loss + dloss

    writer_obj.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer,
                                                model_path=config.model_pth)
    writer_obj.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


# Argument parser (can be outside if __name__ == '__main__': block)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# --- FIX: Changed default exp-name to use valid characters ---
parser.add_argument('--exp-name', type=str, default="pt_ft_rc", help='=exp name')
parser.add_argument('--dataset', type=str, default="others", help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', type=bool, default=True, help='=use layernorm')  # Unused
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--normalization', type=bool, default=True, help='Normalize data')
parser.add_argument('--norm_way', type=str, default='first', help='=use only spatial fc')
parser.add_argument('--rc', type=bool, default=True, help='=use only spatial fc')
parser.add_argument('--permute_p', type=bool, default=True, help='Permute P dimension')
parser.add_argument('--random_rotate', type=bool, default=True, help='Random rotation around world center')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--hd', type=int, default=256, help='=num of blocks')
# Added missing 'global_num' argument (if it's used in config.motion_mlp.num_global_layers)
parser.add_argument('--global_num', type=int, default=4, help='=num of global blocks')
parser.add_argument('--interaction_interval', type=int, default=16,
                    help='Interval between local and Global interactions, must be divisible by num')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--n_p', type=int, default=2)
parser.add_argument('--model_path', type=str, default='pt_ckpts/pt_rc.pth')
parser.add_argument('--vis_every', type=int, default=250000000000, help='Visualize every N iterations')
parser.add_argument('--save_every', type=int, default=100, help='Save model and evaluate every N iterations')
parser.add_argument('--print_every', type=int, default=100, help='Print training info every N iterations')
parser.add_argument('--batch_size', type=int, default=128)
# Added missing 'paixu' argument (if it's used with config.paixu)
parser.add_argument('--paixu', type=bool, default=False, help='Enable person permutation/sorting')
args = parser.parse_args()

# expr_dir definition (can be outside, as it's just a path string)
expr_dir = os.path.join('exprs', args.exp_name)

# Global variables to store latest batch people info
latest_people_stats = {
    'min': 0,
    'max': 0,
    'avg': 0.0,
    'counts': []
}

# ==============================================================================
# All main execution logic MUST be placed inside the if __name__ == '__main__': block
# ==============================================================================
if __name__ == '__main__':
    # --- Close any existing file handlers before attempting directory cleanup ---
    # This is crucial if a previous run failed and left log files open
    close_all_file_handlers_in_dir(expr_dir)

    if os.path.exists(expr_dir):
        print(f"DEBUG: Attempting to remove existing experiment directory: {expr_dir}")
        try:
            shutil.rmtree(expr_dir)
            print(f"DEBUG: Successfully removed {expr_dir}")
        except PermissionError as e:
            print(f"ERROR: Failed to remove directory {expr_dir} due to permission error: {e}")
            print(
                "Please ensure no other programs are using files in this directory, or that file handles are properly closed.")
            print(
                "You may need to manually close any programs that have files open in this directory (e.g., text editors, log viewers), or restart your Python environment.")
            sys.exit(1)  # Exit if cleanup fails
        except OSError as e:
            print(f"ERROR: Failed to remove directory {expr_dir} due to OS error: {e}")
            sys.exit(1)

    os.makedirs(expr_dir, exist_ok=True)

    # Ensure experiment reproducibility
    seed_set(args.seed)
    torch.use_deterministic_algorithms(True)

    # Initialize file handles to None for safe closing in finally block
    acc_log = None
    acc_best_log = None
    writer = None
    logger = None

    try:
        # File for recording metrics
        acc_log_dir = os.path.join(expr_dir, 'acc_log.txt')
        acc_log = open(acc_log_dir, 'a')
        acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))
        acc_log.flush()

        acc_best_log_dir = os.path.join(expr_dir, 'acc_best_log.txt')
        acc_best_log = open(acc_best_log_dir, 'a')
        acc_best_log.write(''.join('Seed : ' + str(args.seed) + '\n'))
        acc_best_log.flush()

        # Configuration
        config.rc = args.rc
        config.norm_way = args.norm_way
        config.normalization = args.normalization
        config.batch_size = args.batch_size
        config.dataset = args.dataset
        config.n_p = args.n_p
        config.vis_every = args.vis_every
        config.save_every = args.save_every
        config.print_every = args.print_every
        config.debug = args.debug
        config.device = args.device
        config.expr_dir = expr_dir
        config.motion_fc_in.temporal_fc = args.temporal_only
        config.motion_fc_out.temporal_fc = args.temporal_only
        config.motion_mlp.norm_axis = args.layer_norm_axis
        config.motion_mlp.spatial_fc_only = args.spatial_fc
        config.motion_mlp.with_normalization = args.with_normalization
        config.motion_mlp.num_layers = args.num
        config.motion_mlp.num_global_layers = args.global_num  # Assuming args.global_num is defined
        config.motion_mlp.n_p = args.n_p
        config.motion_mlp.interaction_interval = args.interaction_interval
        config.motion_mlp.hidden_dim = args.hd
        config.snapshot_dir = os.path.join(expr_dir, 'snapshot')
        ensure_dir(config.snapshot_dir)  # Create folder
        config.vis_dir = os.path.join(expr_dir, 'vis')
        ensure_dir(config.vis_dir)  # Create folder
        config.log_file = os.path.join(expr_dir, 'log.txt')
        # Convert relative path to absolute path using project root
        if args.model_path and not os.path.isabs(args.model_path):
            config.model_pth = path_to_repo(args.model_path)
        else:
            config.model_pth = args.model_path
        config.paixu = args.paixu  # Assuming args.paixu is defined
        # Enable GCN-based approach
        config.motion_mlp.use_gcn = True           # Set to True to use GCN
        config.motion_mlp.gcn_layers = 2           # Number of GCN layers
        
        # Dynamic GCN configuration based on experiment name
        if 'k_' in args.exp_name:
            # k-NN configuration
            k_val = args.exp_name.split('k_')[-1]
            try:
                config.motion_mlp.k_neighbors = int(k_val.split('_')[0])
                config.motion_mlp.distance_threshold = None
                print(f"Using k-NN GCN with k={config.motion_mlp.k_neighbors}")
            except:
                config.motion_mlp.k_neighbors = 2
                config.motion_mlp.distance_threshold = None
        elif 'dis_' in args.exp_name:
            # Distance threshold configuration  
            dis_val = args.exp_name.split('dis_')[-1]
            try:
                config.motion_mlp.distance_threshold = float(dis_val.replace('_', '.'))
                config.motion_mlp.k_neighbors = None
                print(f"Using distance threshold GCN with threshold={config.motion_mlp.distance_threshold}")
            except:
                config.motion_mlp.distance_threshold = 1.5
                config.motion_mlp.k_neighbors = None
        else:
            # Default: Gaussian kernel (full connection)
            config.motion_mlp.k_neighbors = None
            config.motion_mlp.distance_threshold = None
            print("Using Gaussian kernel GCN (full connection)")
        writer = SummaryWriter()

        # Get DCT matrix
        dct_m, idct_m = get_dct_matrix(config.dct_len)
        dct_m = torch.tensor(dct_m).float().to(config.device).unsqueeze(0)
        idct_m = torch.tensor(idct_m).float().to(config.device).unsqueeze(0)
        config.dct_m = dct_m
        config.idct_m = idct_m

        # DataLoader instantiation MUST be inside if __name__ == '__main__':
        if hasattr(config, 'use_mixed_people_dataset') and config.use_mixed_people_dataset:
            config.n_joint = 15  # Ensure n_joint matches MixedPeopleDataset
            config.motion.dim = config.n_joint * 3  # Update motion dim accordingly
            print(" Using Mixed People Dataset for true dynamic people count!")
            data_dir = os.path.join(config.root_dir, 'data')
            mixed_dataset_train = MixedPeopleDataset(data_dir, t_his=config.t_his, t_pred=config.t_pred, split='train')
            mixed_dataset_test = MixedPeopleDataset(data_dir, t_his=config.t_his, t_pred=config.t_pred, split='test')
            
            dataloader_train = torch.utils.data.DataLoader(
                mixed_dataset_train, 
                batch_size=config.batch_size, 
                shuffle=True,
                collate_fn=collate_mixed_batch,
                num_workers=config.num_workers
            )
            dataloader_test = torch.utils.data.DataLoader(
                mixed_dataset_test, 
                batch_size=config.batch_size, 
                shuffle=True,
                collate_fn=collate_mixed_batch,
                num_workers=config.num_workers
            )
            dataloader_test_sample = torch.utils.data.DataLoader(
                mixed_dataset_test, 
                batch_size=1, 
                shuffle=True,
                collate_fn=collate_mixed_batch,
                num_workers=0
            )
        else:
            config.n_joint = 13  # Ensure n_joint matches 3DPW dataset
            config.motion.dim = config.n_joint * 3  # Update motion dim accordingly
            print("Using standard 3DPW dataset")
            dataloader_train = get_3dpw_dataloader(split="train", cfg=config, shuffle=True)
            dataloader_test = get_3dpw_dataloader(split="jrt", cfg=config, shuffle=True)
            dataloader_test_sample = get_3dpw_dataloader(split="jrt", cfg=config, shuffle=True, batch_size=1)
        random_iter = iter(dataloader_test_sample)

        # Create model
        model = Model(config).to(device=config.device)
        model.train()
        print(">>> total params: {:.2f}M".format(
            sum(p.numel() for p in list(model.parameters())) / 1000000.0))

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.cos_lr_max,
                                     weight_decay=config.weight_decay)

        # Create logger
        logger = get_logger(config.log_file, 'train')
        print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True, default=default_serializer))

        # Skip loading pretrained weights for GCN models due to structure changes
        if config.model_pth is not None and not hasattr(config.motion_mlp, 'use_gcn'):
            state_dict = torch.load(config.model_pth, map_location=config.device)
            model.load_state_dict(state_dict, strict=True)
            print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))
            print("Loading model path from {} ".format(config.model_pth))
        else:
            print_and_log_info(logger, "Training GCN model from scratch (pretrained weights skipped)")
            print("Training GCN model from scratch (pretrained weights skipped)")

        ##### ------ Training ------- #####
        nb_iter = 0
        avg_loss = 0.
        avg_lr = 0.
        min_vim = 100000
        metric_best = edict()

        while (nb_iter + 1) < config.cos_lr_total_iters:
            print(f"{nb_iter + 1} / {config.cos_lr_total_iters}")

            for (joints, masks, padding_mask) in dataloader_train:
                # 调试：打印数据格式
                if nb_iter == 0:
                    print(f"DEBUG: joints type: {type(joints)}, shape: {joints.shape if hasattr(joints, 'shape') else 'No shape'}")
                    print(f"DEBUG: masks type: {type(masks)}")
                    print(f"DEBUG: padding_mask type: {type(padding_mask)}, shape: {padding_mask.shape if hasattr(padding_mask, 'shape') else 'No shape'}")
                  
                # B,P,T,J,3 -> B,P,T,JK
                h36m_motion_input = joints[:, :, :config.t_his].flatten(-2)  # 16
                h36m_motion_target = joints[:, :, config.t_his:].flatten(-2)  # 14

                # Ensure tensors are on the correct device and type
                h36m_motion_input = h36m_motion_input.float().to(config.device)
                h36m_motion_target = h36m_motion_target.float().to(config.device)
                padding_mask = padding_mask.to(config.device)  # Move mask to device as well

                loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, padding_mask, model,
                                                         optimizer, nb_iter, config.cos_lr_total_iters,
                                                         config.cos_lr_max, config.cos_lr_min, writer)

                if nb_iter == 9:
                    print("Loss at 10th iteration:", loss)

                avg_loss += loss
                avg_lr += current_lr

                # Print loss
                if (nb_iter + 1) % config.print_every == 0:
                    avg_loss = avg_loss / config.print_every
                    avg_lr = avg_lr / config.print_every

                    print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
                    print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
                    avg_loss = 0
                    avg_lr = 0

                # Save model and evaluate model
                if (nb_iter + 1) % config.save_every == 0 or nb_iter == 0:
                    with torch.no_grad():
                        model.eval()

                        print("begin test")
                        print(f"Iteration: {nb_iter + 1} / {config.cos_lr_total_iters}")
                        
                        # 输出最近一批训练数据的人数统计信息
                        if latest_people_stats['counts']:
                            print(f"Latest batch people stats - Min: {latest_people_stats['min']}, Max: {latest_people_stats['max']}, Avg: {latest_people_stats['avg']:.1f}")

                        mpjpe, vim, jpe, ape, fde = mpjpe_vim_test(config, model, dataloader_test, is_mocap=False,
                                                                   select_vim_frames=[1, 3, 7, 9, 13],
                                                                   select_mpjpe_frames=[7, 14, 14])

                        print(f"iter:{nb_iter}, vim:", vim)
                        print(f"iter:{nb_iter}, mpjpe:", mpjpe)

                        update_metric(metric_best, "vim", vim, nb_iter)
                        update_metric(metric_best, "mpjpe", mpjpe, nb_iter)
                        update_metric(metric_best, "jpe", jpe, nb_iter)
                        update_metric(metric_best, "ape", ape, nb_iter)
                        update_metric(metric_best, "fde", fde, nb_iter)

                        write("vim", vim, nb_iter, acc_log)
                        write("mpjpe", mpjpe, nb_iter, acc_log)
                        write("jpe", jpe, nb_iter, acc_log)
                        write("ape", ape, nb_iter, acc_log)
                        write("fde", fde, nb_iter, acc_log)

                        write("vim", metric_best.vim.val, metric_best.vim.iter, acc_best_log)
                        write("mpjpe", metric_best.mpjpe.val, metric_best.mpjpe.iter, acc_best_log)
                        write("jpe", metric_best.jpe.val, metric_best.jpe.iter, acc_best_log)
                        write("ape", metric_best.ape.val, metric_best.ape.iter, acc_best_log)
                        write("fde", metric_best.fde.val, metric_best.fde.iter, acc_best_log)

                        model.train()

                # Visualize model
                if ((nb_iter + 1) % config.vis_every == 0):
                    model.eval()
                    with torch.no_grad():
                        h36m_motion_input, motion_pred = random_pred(config=config, model=model, iter=random_iter)

                        if h36m_motion_input is not None:
                            b, p, n, c = motion_pred.shape
                            motion_pred = motion_pred.reshape(b, p, n, config.n_joint, 3)
                            h36m_motion_input = h36m_motion_input.reshape(b, p, config.t_his, config.n_joint, 3)
                            motion = torch.cat([h36m_motion_input, motion_pred], dim=2).cpu().detach().numpy()
                            visuaulize(motion, f"iter:{nb_iter}", config.vis_dir, input_len=15, dataset='mupots')

                        model.train()

                if (nb_iter + 1) == config.cos_lr_total_iters:
                    break
                nb_iter += 1

    finally:
        # --- Ensure all file handles are closed when the script exits (normally or abnormally) ---
        print("DEBUG: Script finishing. Closing all open file handles.")
        if acc_log:
            acc_log.close()
            print(f"DEBUG: Closed {acc_log_dir}")
        if acc_best_log:
            acc_best_log.close()
            print(f"DEBUG: Closed {acc_best_log_dir}")

        # Close all logging.FileHandler instances associated with expr_dir
        close_all_file_handlers_in_dir(expr_dir)

        if writer:
            writer.close()
            print("DEBUG: TensorBoard writer closed.")

