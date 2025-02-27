import os
import types
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange

from examples.configs.cdt_configs import CDT_DEFAULT_CONFIG, CDTTrainConfig
from osrl.algorithms import CDT, CDTTrainer
from osrl.common import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all, load_config_and_model

@dataclass
class MergeCDTConfig(CDTTrainConfig):
    # 模型合并相关参数
    use_merge: bool = True  # 是否使用模型合并
    update_steps: int = 100000
    # 已有参数
    cost_model_path: str = "/mnt/e/Users/supergod/Desktop/merge/examples/train/logs/OfflineCarCircle-v0-cost-10/CDT_loss_return_weight0.0_update_steps100000_use_rewFalse-bc70/CDT_loss_return_weight0.0_update_steps100000_use_rewFalse-bc70/checkpoint/model.pt"
    reward_model_path: str = "/mnt/e/Users/supergod/Desktop/merge/examples/train/logs/OfflineCarCircle-v0-cost-10/CDT_loss_cost_weight0.0_update_steps100000_use_costFalse-dbff/CDT_loss_cost_weight0.0_update_steps100000_use_costFalse-dbff/checkpoint/model.pt"
    best: bool = False
    finetune_lr: float = 1e-5
    logdir: str = "logs"
    project: str = "OSRL"
    group: str = "MergedCDT"
    name: str = f"merged_cdt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_path: str = "datasets"
    merge_method: str = "weighted"  # 可选["weighted", "attention", "gated"]

@pyrallis.wrap()
def train_merge(args: MergeCDTConfig):
    # 加载默认配置
    default_cfg = CDT_DEFAULT_CONFIG[args.task]()

    # 更新配置
    cfg = asdict(default_cfg)
    cfg.update(asdict(args))

    # 计算正确的seq_repeat
    seq_repeat = 2  # 基础值
    if cfg["use_cost"]:
        seq_repeat += 1
    if cfg["use_rew"]:
        seq_repeat += 1
    if cfg["cost_prefix"]:
        seq_repeat += 1

    # 设置logger
    if args.logdir is not None:
        # 创建第一层目录
        base_dir = os.path.join(args.logdir, args.group, args.name)
        # 创建第二层同名目录，这是实际存储文件的目录
        run_dir = os.path.join(base_dir, args.name)
        # 创建checkpoint目录在第二层目录下
        checkpoint_dir = os.path.join(run_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 更新args.logdir为第一层目录，这样WandbLogger会在正确的位置创建文件
        args.logdir = base_dir

    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg)

    def checkpoint_fn():
        return {
            "model_state": model.state_dict(),
            "args": cfg,
        }

    # 设置随机种子
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # 初始化环境
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    # 预处理数据集
    data = env.pre_process_data(data)
    env = wrap_env(env=env, reward_scale=args.reward_scale)
    env = OfflineEnvWrapper(env)

    # 创建merge模型
    model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=args.embedding_dim,
        seq_len=args.seq_len,
        episode_len=args.episode_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        residual_dropout=args.residual_dropout,
        embedding_dropout=args.embedding_dropout,
        time_emb=args.time_emb,
        use_rew=True,
        use_cost=True,
        cost_transform=args.cost_transform,
        add_cost_feat=args.add_cost_feat,
        mul_cost_feat=args.mul_cost_feat,
        cat_cost_feat=args.cat_cost_feat,
        action_head_layers=args.action_head_layers,
        cost_prefix=args.cost_prefix,
        stochastic=args.stochastic,
        init_temperature=args.init_temperature,
        target_entropy=-env.action_space.shape[0],
    ).to(args.device)

    # 加载预训练模型，使用指定的merge方法
    model.load_pretrained_models(
        args.reward_model_path,
        args.cost_model_path,
        merge_method=args.merge_method
    )

    # 创建数据集和dataloader
    dataset = SequenceDataset(
        data,
        seq_len=args.seq_len,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
    )
    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    # 创建trainer
    trainer = CDTTrainer(
        model,
        env,
        logger=logger,
        learning_rate=args.finetune_lr,  # 使用finetune的学习率
        weight_decay=args.weight_decay,
        betas=args.betas,
        clip_grad=args.clip_grad,
        lr_warmup_steps=args.lr_warmup_steps,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        loss_cost_weight=args.loss_cost_weight,
        loss_state_weight=args.loss_state_weight,
        loss_return_weight=args.loss_return_weight,
        cost_reverse=args.cost_reverse,
        no_entropy=args.no_entropy,
        device=args.device,
    )

    # 训练循环
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = [
            b.to(args.device) for b in batch
        ]

        # 训练步骤
        trainer.train_one_step(states, actions, returns, costs_return, time_steps, mask,
                               episode_cost, costs)

        # 评估和checkpoint保存逻辑
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            average_reward, average_cost = [], []
            log_cost, log_reward, log_len = {}, {}, {}

            for target_return in args.target_returns:
                reward_return, cost_return = target_return
                ret, cost, length = trainer.evaluate(args.eval_episodes,
                                                    reward_return * args.reward_scale,
                                                    cost_return * args.cost_scale)
                average_cost.append(cost)
                average_reward.append(ret)

                name = "c_" + str(int(cost_return)) + "_r_" + str(int(reward_return))
                log_cost.update({name: cost})
                log_reward.update({name: ret})
                log_len.update({name: length})

            logger.store(tab="cost", **log_cost)
            logger.store(tab="ret", **log_reward)
            logger.store(tab="length", **log_len)

            # 保存当前权重到第二层目录的checkpoint文件夹
            torch.save(checkpoint_fn(), os.path.join(checkpoint_dir, "model.pt"))

            # 保存最佳权重到第二层目录的checkpoint文件夹
            mean_ret = np.mean(average_reward)
            mean_cost = np.mean(average_cost)
            if mean_cost < best_cost or (mean_cost == best_cost and mean_ret > best_reward):
                best_cost = mean_cost
                best_reward = mean_ret
                best_idx = step
                torch.save(checkpoint_fn(), os.path.join(checkpoint_dir, "model_best.pt"))

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)
        else:
            logger.write_without_reset(step)

    print(f"Training finished. Models saved in {checkpoint_dir}")
    print(f"Best model achieved at step {best_idx} with cost {best_cost:.2f} and reward {best_reward:.2f}")

if __name__ == "__main__":
    train_merge() 