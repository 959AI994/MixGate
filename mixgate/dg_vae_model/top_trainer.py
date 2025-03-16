from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import time
from torch import nn
from progress.bar import Bar
from torch_geometric.loader import DataLoader

from .utils.utils import zero_normalization, AverageMeter
from .utils.logger import Logger

class TopTrainer():
    def __init__(self, args, model, device='cpu', distributed=False):
        super(TopTrainer, self).__init__()
        self.args = args
        self.device = device
        self.distributed = distributed
        
        # 初始化配置
        self._init_training_env(args)
        self._init_model(model)
        self._init_loss_functions()
        self._init_optimizer()
        
        # 日志记录器
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
        
    def _init_training_env(self, args):
        """初始化训练环境"""
        self.lr = args.lr
        self.lr_step = args.lr_step
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        # 创建保存目录
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        self.log_dir = os.path.join(args.save_dir, args.exp_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # 分布式训练配置
        self.local_rank = 0
        if self.distributed and torch.cuda.is_available():
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print(f'Distributed training on device {self.device}, rank {self.rank}/{self.world_size}')
        else:
            print(f'Single device training on {self.device}')

    def _init_model(self, model):
        """初始化模型和设备"""
        self.model = model.to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
        self.model_epoch = 0

    def _init_loss_functions(self):
        """初始化损失函数"""
        self.reg_loss = nn.L1Loss().to(self.device)
        self.clf_loss = nn.BCELoss().to(self.device)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.cos_sim = nn.CosineEmbeddingLoss().to(self.device)

    def _init_optimizer(self):
        """初始化优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

    def compute_loss(self, outputs, batch):
        """多任务损失计算"""
        # 重建损失
        mask = outputs['mask_indices']
        recon_loss = self.clf_loss(
            outputs['pred_prob'], 
            batch.y[mask].float().unsqueeze(1)
        )
        
        # KL散度损失（从各子模型收集）
        kl_loss = 0
        for model in [self.model.module.deepgate_aig, 
                     self.model.module.deepgate_mig,
                     self.model.module.deepgate_xmg,
                     self.model.module.deepgate_xag]:
            mu = model.vae_mu.buffer
            logvar = model.vae_logstd.buffer
            kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 功能相似性损失
        emb_dis = 1 - torch.cosine_similarity(
            outputs['reconstructed'][batch.tt_pair_index[0]],
            outputs['reconstructed'][batch.tt_pair_index[1]],
            dim=1
        )
        func_loss = self.reg_loss(
            zero_normalization(emb_dis),
            zero_normalization(batch.tt_dis)
        )
        
        # 子模型对齐损失
        align_loss = self._compute_alignment_loss(outputs, batch)
        
        # 总损失
        total_loss = (
            recon_loss * self.args.alpha +
            kl_loss * self.args.beta +
            func_loss * self.args.gamma +
            align_loss * self.args.delta
        )
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss,
            'func': func_loss,
            'align': align_loss
        }

    def _compute_alignment_loss(self, outputs, batch):
        """多模态对齐损失"""
        loss = 0
        modalities = ['aig', 'mig', 'xmg', 'xag']
        for mod in modalities:
            pred = outputs[f'{mod}_prob']
            target = getattr(batch, f'{mod}_prob')
            loss += self.reg_loss(pred, target.unsqueeze(1))
        return loss / len(modalities)

    def train_epoch(self, dataloader, epoch, phase='train'):
        """单epoch训练"""
        batch_time = AverageMeter()
        loss_stats = {k: AverageMeter() for k in ['total', 'recon', 'kl', 'func', 'align']}
        
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
            torch.cuda.empty_cache()

        bar = Bar(f'{phase} Epoch {epoch}', max=len(dataloader)) if self.local_rank == 0 else None
        
        for iter_id, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            start_time = time.time()
            
            # 前向传播
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(batch)
                loss_dict = self.compute_loss(outputs, batch)
                
            # 反向传播
            if phase == 'train':
                self.optimizer.zero_grad()
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
            # 更新统计量
            batch_time.update(time.time() - start_time)
            for k in loss_stats:
                loss_stats[k].update(loss_dict[k].item())
                
            # 进度条更新
            if self.local_rank == 0:
                bar.suffix  = f'[{iter_id}/{len(dataloader)}]|'
                bar.suffix += f'Time:{batch_time.avg:.1f}s|'
                bar.suffix += f'Total:{loss_stats["total"].avg:.3f}|'
                bar.suffix += f'Recon:{loss_stats["recon"].avg:.3f}|'
                bar.suffix += f'KL:{loss_stats["kl"].avg:.3f}|'
                bar.suffix += f'Func:{loss_stats["func"].avg:.3f}|'
                bar.suffix += f'Align:{loss_stats["align"].avg:.3f}'
                bar.next()
                
        if self.local_rank == 0:
            bar.finish()
            self.logger.write(
                f'{phase}|Epoch {epoch}|'
                f'Total:{loss_stats["total"].avg:.3f}|'
                f'Recon:{loss_stats["recon"].avg:.3f}|'
                f'KL:{loss_stats["kl"].avg:.3f}|'
                f'Func:{loss_stats["func"].avg:.3f}|'
                f'Align:{loss_stats["align"].avg:.3f}\n'
            )
            
        return loss_stats['total'].avg

    def train(self, num_epochs, train_dataset, val_dataset):
        """完整训练流程"""
        train_loader = self._init_loader(train_dataset, shuffle=True)
        val_loader = self._init_loader(val_dataset, shuffle=False)
        
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = self.train_epoch(train_loader, epoch, 'train')
            
            # 验证阶段
            with torch.no_grad():
                val_loss = self.train_epoch(val_loader, epoch, 'val')
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss and self.local_rank == 0:
                best_val_loss = val_loss
                self.save_model('model_best.pth')
                
            # 定期保存
            if epoch % 10 == 0 and self.local_rank == 0:
                self.save_model(f'model_{epoch}.pth')
                
    def _init_loader(self, dataset, shuffle):
        """初始化数据加载器"""
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=shuffle,
                num_replicas=self.world_size,
                rank=self.rank
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers
            )

    def save_model(self, filename):
        """保存模型"""
        state = {
            'epoch': self.model_epoch,
            'state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.scheduler.best
        }
        torch.save(state, os.path.join(self.log_dir, filename))

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.model_epoch = checkpoint['epoch']
        print(f'Loaded checkpoint from epoch {self.model_epoch}')
