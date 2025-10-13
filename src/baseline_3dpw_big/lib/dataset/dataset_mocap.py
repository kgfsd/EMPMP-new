import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from io import BytesIO
from torch.utils.data import Dataset
from copy import deepcopy as c
import torch_dct as dct
class DATA(Dataset):
    def __init__(self, mode, t_his=15, t_pred=45,use_v=False,n_p=3, data_root=None):
        # 如果没有指定data_root，尝试找到项目根目录
        if data_root is None:
            # 从当前文件位置向上查找，直到找到data目录
            current_file = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file)
            # 向上查找直到找到包含data目录的路径
            for _ in range(5):  # 最多向上查找5层
                parent_dir = os.path.dirname(current_dir)
                data_dir = os.path.join(parent_dir, 'data')
                if os.path.exists(data_dir):
                    data_root = data_dir
                    break
                current_dir = parent_dir
            
            # 如果还没找到，使用相对路径
            if data_root is None:
                data_root = './data'
        
        if mode=="train":
            self.data=np.load(os.path.join(data_root, 'train_3_120_mocap.npy'),allow_pickle=True)
        elif mode=="test":
            self.data=np.load(os.path.join(data_root, 'test_3_120_mocap.npy'),allow_pickle=True)
        elif mode=="eval_mutpots":
            self.data=np.load(os.path.join(data_root, 'mupots_120_3persons.npy'),allow_pickle=True)
        elif mode=="eval_mocap":
            self.data=np.load(os.path.join(data_root, 'test_3_120_mocap.npy'),allow_pickle=True)

        # #[n,P,T(120),J,3]
        # self.data=self.data.reshape(self.data.shape[0],self.data.shape[1],self.data.shape[2],-1,3)
        #单人
        if n_p==1:
            self.data=self.data.reshape(self.data.shape[0],self.data.shape[1],self.data.shape[2],-1)[:,:1,:,:]

        self.traj_dim = 15 * 3
        if use_v==True:
            self.traj_dim = 16 * 3
            v = (np.diff(self.data[:,:,:, :1,:], axis=2) * 50).clip(-5.0, 5.0)#计算vel只需要一个关节的信息
            v = np.append(v, v[:,:,-1:], axis=2)
            self.data=np.concatenate((self.data, v), axis=3)
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.std, self.mean = None, None
        
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        
    def sample(self):        
        i=np.random.choice(list(range(0,len(self.data))))
        seq=self.data[i]
        traj = seq[:,::2,:]
        if self.t_total!=60:
            start_idx = np.random.choice(range(0, traj.shape[1] - self.t_total))
            traj = traj[:,start_idx:start_idx+self.t_total,:]
        return traj[None, ...]
    


    def sampling_generator(self, num_samples=10000, batch_size=64, aug=True):
        for i in range(num_samples // batch_size):
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            yield sample[:,:,:self.t_his],sample[:,:,self.t_his:]
            
            
    def iter_generator(self,  batch_size=128):#复现MRT,120帧间隔取60帧
        total_batches = len(self.data) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            data = self.data[start_idx:start_idx + batch_size, :, ::2]
            yield data[:, :, :self.t_his], data[:, :, self.t_his:]
        
        # 处理最后一小部分数据
        if len(self.data) % batch_size != 0:
            start_idx = total_batches * batch_size
            data = self.data[start_idx:, :, ::2]  # 最后的小部分数据
            yield data[:, :, :self.t_his], data[:, :, self.t_his:]
            
    def __getitem__(self, idx):
        # 使用配置的 t_his 和 t_pred，而不是硬编码的50
        data = self.data[idx][:,::2,:]  # [P, T, JK]
        return data[:,:self.t_his], data[:,self.t_his:self.t_his+self.t_pred]
    def __len__(self):
        return len(self.data)

def get_mocap_dataloader_for_mixed(split="test", t_his=16, t_pred=14, batch_size=128, shuffle=False, n_p=3, data_root=None):
    """
    创建与混合数据集格式兼容的mocap数据加载器
    返回格式: (joints, masks, padding_mask)
    joints: [B, P, T, J, 3]
    padding_mask: [B, P] (bool)
    
    Args:
        data_root: 数据目录的绝对路径，如果为None则自动查找
    """
    # 加载mocap数据集
    mocap_dataset = DATA(mode=split, t_his=t_his, t_pred=t_pred, use_v=False, n_p=n_p, data_root=data_root)
    
    def collate_mocap_to_mixed_format(batch):
        """将mocap数据转换为混合数据集格式"""
        # batch中每个元素是 (input, target)
        # input: [P, T_his, JK], target: [P, T_pred, JK]
        batch_inputs = []
        batch_targets = []
        
        for input_data, target_data in batch:
            # 合并input和target: [P, T_total, JK]
            combined = np.concatenate([input_data, target_data], axis=1)
            batch_inputs.append(combined)
        
        # 转换为numpy数组
        batch_data = np.array(batch_inputs)  # [B, P, T_total, JK]
        
        # 重塑为 [B, P, T, J, 3]
        B, P, T, JK = batch_data.shape
        J = JK // 3
        batch_data = batch_data.reshape(B, P, T, J, 3)
        
        # 创建padding mask (mocap数据集所有人都是有效的)
        padding_mask = np.ones((B, P), dtype=bool)
        
        # 转换为torch张量
        joints = torch.FloatTensor(batch_data)
        masks = None  # mocap不需要masks
        padding_mask = torch.BoolTensor(padding_mask)
        
        return joints, masks, padding_mask
    
    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        mocap_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_mocap_to_mixed_format,
        num_workers=0  # mocap数据集使用0个worker
    )
    
    return dataloader

def main():
    # 创建训练集实例
    train_dataset = DATA(mode="train", t_his=15, t_pred=45,n_p=3)
    
    print(f"训练集大小: {len(train_dataset)}")
    

    # 使用DataLoader
    generator=train_dataset.iter_generator(batch_size=1)
    for (batch,a) in generator:
        print(f"DataLoader批次形状: {batch.shape}")
        break
    
def visuaulize(data,prefix,output_dir):
    for n in range(data.shape[0]):
        data_list=data[n]
        body_edges = np.array(
            [[0, 1], [1, 2], [2, 3], [0, 4],
            [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
        )

        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        # 创建保存帧的列表
        frames = []
        frame_names=[]
        length_ = data_list.shape[1]

        for i in range(0, length_):
            ax.cla()  # Clear the previous lines
            for j in range(len(data_list)):
                xs = data_list[j, i, :, 0]
                ys = data_list[j, i, :, 1]
                zs = data_list[j, i, :, 2]
                ax.plot(zs, xs, ys, 'y.')
                
                plot_edge = True
                if plot_edge:
                    for edge in body_edges:
                        x = [data_list[j, i, edge[0], 0], data_list[j, i, edge[1], 0]]
                        y = [data_list[j, i, edge[0], 1], data_list[j, i, edge[1], 1]]
                        z = [data_list[j, i, edge[0], 2], data_list[j, i, edge[1], 2]]
                        if i >= 15:
                            ax.plot(z, x, y, 'green')
                        else:
                            ax.plot(z, x, y, 'blue')
                
                ax.set_xlim3d([-2, 2])
                ax.set_ylim3d([-2, 2])
                ax.set_zlim3d([-0, 2])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            
            # 保存当前帧到图像
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            i += 1

        # 保存为GIF
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imageio.mimsave(f'./{output_dir}/{prefix}_{n}.gif', frames, duration=0.1)


        # 清理临时帧图像
        for frame_filename in frame_names:
            # print(frame_filename)
            os.remove(frame_filename)

if __name__ == "__main__":
    main()