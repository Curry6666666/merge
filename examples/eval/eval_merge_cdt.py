
import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym
import numpy as np
import pyrallis
import torch
from dataclasses import asdict, dataclass
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from examples.configs.eval_configs import EvalConfig
from osrl.algorithms import CDT, CDTTrainer
from osrl.common.exp_util import load_config_and_model, seed_all

@dataclass
class MergeEvalConfig(EvalConfig):
    merge_model_path: str = "path/to/merge/model.pt"

@pyrallis.wrap()
def eval_merge(args: MergeEvalConfig):
    # ����mergeģ�����ú�Ȩ��
    cfg, model = load_config_and_model(args.merge_model_path, args.best)
    
    # �����������
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # ��������
    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_limit"])

    # ����������mergeģ��
    cdt_model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=cfg["embedding_dim"],
        seq_len=cfg["seq_len"],
        episode_len=cfg["episode_len"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        attention_dropout=cfg["attention_dropout"],
        residual_dropout=cfg["residual_dropout"],
        embedding_dropout=cfg["embedding_dropout"],
        time_emb=cfg["time_emb"],
        use_rew=True,
        use_cost=True,
        cost_transform=cfg["cost_transform"],
        add_cost_feat=cfg["add_cost_feat"],
        mul_cost_feat=cfg["mul_cost_feat"],
        cat_cost_feat=cfg["cat_cost_feat"],
        action_head_layers=cfg["action_head_layers"],
        cost_prefix=cfg["cost_prefix"],
        stochastic=cfg["stochastic"],
        init_temperature=cfg["init_temperature"],
        target_entropy=-env.action_space.shape[0],
    ).to(args.device)
    
    cdt_model.load_state_dict(model["model_state"])
    
    # ����trainer��������
    trainer = CDTTrainer(
        cdt_model,
        env,
        reward_scale=cfg["reward_scale"],
        cost_scale=cfg["cost_scale"],
        cost_reverse=cfg["cost_reverse"],
        device=args.device
    )
    
    # ������ͬĿ��return��cost�µ�����
    for target_ret, target_cost in zip(args.returns, args.costs):
        ret, cost, length = trainer.evaluate(
            args.eval_episodes,
            target_ret * cfg["reward_scale"],
            target_cost * cfg["cost_scale"]
        )
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        print(
            f"Target return {target_ret}, real return {ret}, normalized return: {normalized_ret}; "
            f"target cost {target_cost}, real cost {cost}, normalized cost: {normalized_cost}"
        )

if __name__ == "__main__":
    eval_merge() 