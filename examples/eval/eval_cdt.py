# -*- coding: UTF-8 -*-
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field

from osrl.algorithms import CDT, CDTTrainer
from osrl.common.exp_util import load_config_and_model, seed_all


@dataclass
class EvalConfig:
    # 模型检查点路径
    # path: str = "log/.../checkpoint/model.pt"
    path: str = "examples/train/logs/OfflineCarCircle-v0-cost-10/CDT_loss_return_weight0.0_update_steps100000_use_rewFalse-bc70/CDT_loss_return_weight0.0_update_steps100000_use_rewFalse-bc70"
    # 期望的回报值列表,用于评估不同回报目标下的性能
    returns: List[float] = field(default=[300, 400, 500], is_mutable=True)
    # 期望的成本值列表,用于评估不同成本约束下的性能
    costs: List[float] = field(default=[10, 10, 10], is_mutable=True)
    # 噪声尺度列表,用于评估模型在不同噪声水平下的鲁棒性
    noise_scale: List[float] = None
    # 评估的回合数
    eval_episodes: int = 20
    # 是否使用最佳模型进行评估
    best: bool = False
    # 运行设备,可选CPU或GPU
    device: str = "cpu"
    # CPU线程数
    threads: int = 4


@pyrallis.wrap()
def eval(args: EvalConfig):
    """评估CDT模型的主函数

    Args:
        args (EvalConfig): 评估配置参数
    """
    # 加载模型配置和权重
    cfg, model = load_config_and_model(args.path, args.best)
    # 设置随机种子
    seed_all(cfg["seed"])
    # 如果使用CPU,设置线程数
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # 根据任务名称选择合适的gym环境
    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    # 创建并包装环境
    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )
    # 使用离线环境包装器
    env = OfflineEnvWrapper(env)
    # 设置目标成本限制
    env.set_target_cost(cfg["cost_limit"])

    # 设置目标熵值为动作空间维度的负值
    target_entropy = -env.action_space.shape[0]

    # model & optimizer & scheduler setup
    # 创建CDT模型实例
    cdt_model = CDT(
        state_dim=env.observation_space.shape[0],  # 状态空间维度
        action_dim=env.action_space.shape[0],  # 动作空间维度
        max_action=env.action_space.high[0],  # 动作空间最大值
        embedding_dim=cfg["embedding_dim"],  # 嵌入维度
        seq_len=cfg["seq_len"],  # 序列长度
        episode_len=cfg["episode_len"],  # episode长度
        num_layers=cfg["num_layers"],  # transformer层数
        num_heads=cfg["num_heads"],  # 注意力头数
        attention_dropout=cfg["attention_dropout"],  # 注意力dropout率
        residual_dropout=cfg["residual_dropout"],  # 残差连接dropout率
        embedding_dropout=cfg["embedding_dropout"],  # 嵌入层dropout率
        time_emb=cfg["time_emb"],  # 是否使用时间嵌入
        use_rew=cfg["use_rew"],  # 是否使用奖励信息
        use_cost=cfg["use_cost"],  # 是否使用成本信息
        cost_transform=cfg["cost_transform"],  # 成本变换方式
        add_cost_feat=cfg["add_cost_feat"],  # 是否添加成本特征
        mul_cost_feat=cfg["mul_cost_feat"],  # 是否乘以成本特征
        cat_cost_feat=cfg["cat_cost_feat"],  # 是否拼接成本特征
        action_head_layers=cfg["action_head_layers"],  # 动作头层数
        cost_prefix=cfg["cost_prefix"],  # 是否使用成本前缀
        stochastic=cfg["stochastic"],  # 是否使用随机策略
        init_temperature=cfg["init_temperature"],  # 初始温度参数
        target_entropy=target_entropy,  # 目标熵值
    )
    # 加载预训练模型参数
    cdt_model.load_state_dict(model["model_state"])
    # 将模型迁移到指定设备
    cdt_model.to(args.device)

    # 创建CDT训练器实例
    trainer = CDTTrainer(cdt_model,
                         env,
                         reward_scale=cfg["reward_scale"],  # 奖励缩放系数
                         cost_scale=cfg["cost_scale"],  # 成本缩放系数
                         cost_reverse=cfg["cost_reverse"],  # 是否反转成本
                         device=args.device)  # 运行设备

    # 获取目标奖励和成本列表
    rets = args.returns
    costs = args.costs
    # 确保奖励和成本列表长度相等
    assert len(rets) == len(
        costs
    ), f"The length of returns {len(rets)} should be equal to costs {len(costs)}!"
    # 对每一组目标奖励和成本进行评估
    for target_ret, target_cost in zip(rets, costs):
        # 设置随机种子
        seed_all(cfg["seed"])
        # 评估模型性能
        ret, cost, length = trainer.evaluate(args.eval_episodes,
                                             target_ret * cfg["reward_scale"],
                                             target_cost * cfg["cost_scale"])
        # 获取归一化后的奖励和成本
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        # 打印评估结果
        print(
            f"Target reward {target_ret}, real reward {ret}, normalized reward: {normalized_ret}; target cost {target_cost}, real cost {cost}, normalized cost: {normalized_cost}"
        )


if __name__ == "__main__":
    eval()
