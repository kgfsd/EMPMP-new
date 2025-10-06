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
# Corrected import for 'paixu_person' if it's used with config.paixu
from src.models_dual_inter_traj_big.utils import Get_RC_Data, visuaulize, visuaulize_bianhao, seed_set, get_dct_matrix, \
    gen_velocity, predict, update_metric, getRandomPermuteOrder, \
    getRandomRotatePoseTransform  # Removed paixu_person as it's not in utils in the original code, assuming it's in model if used.
# If paixu_person is indeed in utils, please add it back:
# from src.models_dual_inter_traj_big.utils import Get_RC_Data,visuaulize,visuaulize_bianhao,seed_set,get_dct_matrix,gen_velocity,predict,update_metric,getRandomPermuteOrder,getRandomRotatePoseTransform,paixu_person

from lr import update_lr_multistep
from src.baseline_3dpw_big.config import config
from src.models_dual_inter_traj_big.model import siMLPe as Model
from src.baseline_3dpw_big.lib.dataset.dataset_3dpw import get_3dpw_dataloader
from src.baseline_3dpw_big.lib.utils.logger import get_logger, print_and_log_info
from src.baseline_3dpw_big.lib.utils.pyt_utils import ensure_dir
import torch
from torch.utils.tensorboard import SummaryWriter
from src.baseline_3dpw_big.test import vim_test, random_pred, mpjpe_vim_test

# If inverse_sort_tensor is used, ensure it's imported (it was in your previous version, but not this one)
# from src.models_dual_inter_traj_big.model import inverse_sort_tensor

# Ignore all warnings
warnings.filterwarnings("ignore")


# --- Auxiliary functions (can be outside if __name__ == '__main__': block) ---

def close_all_file_handlers_in_dir(target_dir):
    """
    Closes all logging.FileHandlers if their log files are within target_dir,
    and removes them from their respective loggers.
    """
    abs_target_dir = os.path.abspath(target_dir)

    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name == 'root':
            continue

        logger = logging.getLogger(logger_name)
        handlers_to_remove = []
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler_file_path = handler.baseFilename
                abs_handler_file_path = os.path.abspath(handler_file_path)

                if abs_handler_file_path.startswith(abs_target_dir + os.sep) or abs_handler_file_path == abs_target_dir:
                    print(f"DEBUG: Closing logging.FileHandler for: {handler_file_path}")
                    handler.close()
                    handlers_to_remove.append(handler)

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
               min_lr):
    # Note: If paixu_person and inverse_sort_tensor are used,
    # ensure they are imported and the args.paixu is added to parser.
    # For now, commenting out as they were not in the provided imports.
    # sorted_indices = None
    # if config.paixu:
    #     h36m_motion_input, h36m_motion_target, sorted_indices = paixu_person(h36m_motion_input, h36m_motion_target)

    if args.random_rotate:
        h36m_motion_input, h36m_motion_target = getRandomRotatePoseTransform(h36m_motion_input, h36m_motion_target)
    if args.permute_p:
        h36m_motion_input, h36m_motion_target = getRandomPermuteOrder(h36m_motion_input, h36m_motion_target)
    if config.rc:
        h36m_motion_input, h36m_motion_target = Get_RC_Data(h36m_motion_input, h36m_motion_target)

    motion_pred = predict(model, h36m_motion_input, config, h36m_motion_target)  # b,p,n,c
    b, p, n, c = h36m_motion_target.shape

    # if config.paixu and sorted_indices is not None:
    #     motion_pred = inverse_sort_tensor(motion_pred, sorted_indices)
    #     h36m_motion_target = inverse_sort_tensor(h36m_motion_target, sorted_indices)

    # Predicted pose
    motion_pred = motion_pred.reshape(b, p, n, config.n_joint, 3).reshape(-1, 3)
    # GT
    h36m_motion_target = h36m_motion_target.to(config.device).reshape(b, p, n, config.n_joint, 3).reshape(-1, 3)
    # mask:b,p
    expanded_mask = padding_mask.unsqueeze(-1).repeat(1, 1, n * config.n_joint).reshape(-1)
    # Calculate loss
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, dim=1)[expanded_mask])

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b, p, n, config.n_joint, 3)
        dmotion_pred = gen_velocity(motion_pred)  # Calculate velocity

        motion_gt = h36m_motion_target.reshape(b, p, n, config.n_joint, 3)
        dmotion_gt = gen_velocity(motion_gt)  # Calculate velocity

        expanded_mask = padding_mask.unsqueeze(-1).repeat(1, 1, (n - 1) * config.n_joint).reshape(-1)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), dim=1)[expanded_mask])
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer,
                                                model_path=config.model_pth)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


# Argument parser (can be outside if __name__ == '__main__': block)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# --- FIX: Changed default exp-name to use valid characters ---
parser.add_argument('--exp-name', type=str, default="pt_ft_norc", help='=exp name')
parser.add_argument('--dataset', type=str, default="others", help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', type=bool, default=True, help='=use layernorm')  # Unused
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--normalization', type=bool, default=True, help='Normalize data')
parser.add_argument('--norm_way', type=str, default='first', help='=use only spatial fc')
parser.add_argument('--rc', type=bool, default=False, help='=use only spatial fc')
parser.add_argument('--permute_p', type=bool, default=True, help='Permute P dimension')
parser.add_argument('--random_rotate', type=bool, default=True, help='Random rotation around world center')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--hd', type=int, default=128, help='=num of blocks')
# Added missing 'global_num' argument (if it's used in config.motion_mlp.num_global_layers)
parser.add_argument('--global_num', type=int, default=4, help='=num of global blocks')
parser.add_argument('--interaction_interval', type=int, default=16,
                    help='Interval between local and Global interactions, must be divisible by num')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--n_p', type=int, default=2)
parser.add_argument('--model_path', type=str, default='pt_ckpts/pt_norc.pth')
parser.add_argument('--vis_every', type=int, default=250000000000)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
# Added missing 'paixu' argument (if it's used with config.paixu)
parser.add_argument('--paixu', type=bool, default=False, help='Enable person permutation/sorting')
args = parser.parse_args()

# expr_dir definition (can be outside, as it's just a path string)
expr_dir = os.path.join('exprs', args.exp_name)

# ==============================================================================
# All main execution logic MUST be placed inside the if __name__ == '__main__': block
# ==============================================================================
if __name__ == '__main__':
    # --- Close any existing file handlers before attempting directory cleanup ---
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
        config.model_pth = args.model_path
        config.paixu = args.paixu  # Assuming args.paixu is defined

        writer = SummaryWriter()

        # Get DCT matrix
        dct_m, idct_m = get_dct_matrix(config.dct_len)
        dct_m = torch.tensor(dct_m).float().to(config.device).unsqueeze(0)
        idct_m = torch.tensor(idct_m).float().to(config.device).unsqueeze(0)
        config.dct_m = dct_m
        config.idct_m = idct_m

        # Create model
        model = Model(config).to(device=config.device)
        model.train()
        print(">>> total params: {:.2f}M".format(
            sum(p.numel() for p in list(model.parameters())) / 1000000.0))

        # DataLoader instantiation MUST be inside if __name__ == '__main__':
        dataloader_train = get_3dpw_dataloader(split="train", cfg=config, shuffle=True)
        dataloader_test = get_3dpw_dataloader(split="jrt", cfg=config, shuffle=True)
        dataloader_test_sample = get_3dpw_dataloader(split="jrt", cfg=config, shuffle=True, batch_size=1)
        random_iter = iter(dataloader_test_sample)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.cos_lr_max,
                                     weight_decay=config.weight_decay)

        # Create logger
        logger = get_logger(config.log_file, 'train')
        print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True, default=default_serializer))

        if config.model_pth is not None:  # For pre-training
            state_dict = torch.load(config.model_pth, map_location=config.device)
            model.load_state_dict(state_dict, strict=True)
            print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))
            print("Loading model path from {} ".format(config.model_pth))

        ##### ------ Training ------- #####
        nb_iter = 0
        avg_loss = 0.
        avg_lr = 0.
        min_vim = 100000
        metric_best = edict()

        while (nb_iter + 1) < config.cos_lr_total_iters:
            print(f"{nb_iter + 1} / {config.cos_lr_total_iters}")

            for (joints, masks, padding_mask) in dataloader_train:
                # B,P,T,JK
                h36m_motion_input = joints[:, :, :config.t_his].flatten(-2)  # 16
                h36m_motion_target = joints[:, :, config.t_his:].flatten(-2)  # 14

                h36m_motion_input = torch.tensor(h36m_motion_input).float()
                h36m_motion_target = torch.tensor(h36m_motion_target).float()

                loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, padding_mask, model,
                                                         optimizer, nb_iter, config.cos_lr_total_iters,
                                                         config.cos_lr_max, config.cos_lr_min)

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

