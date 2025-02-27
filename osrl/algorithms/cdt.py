from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from torch.distributions.beta import Beta
from torch.nn import functional as F
from tqdm.auto import trange

from osrl.common.net import DiagGaussianActor, TransformerBlock, mlp
from dataclasses import dataclass, field

class CDT(nn.Module):
    """
    Constrained Decision Transformer (CDT)

    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        seq_len (int): The length of the sequence to process.
        episode_len (int): The length of the episode.
        embedding_dim (int): The dimension of the embeddings.
        num_layers (int): The number of transformer layers to use.
        num_heads (int): The number of heads to use in the multi-head attention.
        attention_dropout (float): The dropout probability for attention layers.
        residual_dropout (float): The dropout probability for residual layers.
        embedding_dropout (float): The dropout probability for embedding layers.
        time_emb (bool): Whether to include time embeddings.
        use_rew (bool): Whether to include return embeddings.
        use_cost (bool): Whether to include cost embeddings.
        cost_transform (bool): Whether to transform the cost values.
        add_cost_feat (bool): Whether to add cost features.
        mul_cost_feat (bool): Whether to multiply cost features.
        cat_cost_feat (bool): Whether to concatenate cost features.
        action_head_layers (int): The number of layers in the action head.
        cost_prefix (bool): Whether to include a cost prefix.
        stochastic (bool): Whether to use stochastic actions.
        init_temperature (float): The initial temperature value for stochastic actions.
        target_entropy (float): The target entropy value for stochastic actions.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            seq_len: int = 10,
            episode_len: int = 1000,
            embedding_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            attention_dropout: float = 0.0,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.0,
            time_emb: bool = True,
            use_rew: bool = False,
            use_cost: bool = False,
            cost_transform: bool = False,
            add_cost_feat: bool = False,
            mul_cost_feat: bool = False,
            cat_cost_feat: bool = False,
            action_head_layers: int = 1,
            cost_prefix: bool = False,
            stochastic: bool = False,
            init_temperature=0.1,
            target_entropy=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        if cost_transform:
            self.cost_transform = lambda x: 50 - x
        else:
            self.cost_transform = None
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat
        self.cat_cost_feat = cat_cost_feat
        self.stochastic = stochastic
        self.use_rew = use_rew
        self.use_cost = use_cost

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        self.time_emb = time_emb
        if self.time_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)

        self.seq_repeat = 2
        if self.use_cost:
            self.cost_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_rew:
            self.return_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1

        dt_seq_len = self.seq_repeat * seq_len

        self.cost_prefix = cost_prefix
        if self.cost_prefix:
            self.prefix_emb = nn.Linear(1, embedding_dim)
            dt_seq_len += 1

        self.blocks = nn.ModuleList([
            TransformerBlock(
                seq_len=dt_seq_len,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
            ) for _ in range(num_layers)
        ])

        action_emb_dim = 2 * embedding_dim if self.cat_cost_feat else embedding_dim

        if self.stochastic:
            if action_head_layers >= 2:
                self.action_head = nn.Sequential(
                    nn.Linear(action_emb_dim, action_emb_dim), nn.GELU(),
                    DiagGaussianActor(action_emb_dim, action_dim))
            else:
                self.action_head = DiagGaussianActor(action_emb_dim, action_dim)
        else:
            self.action_head = mlp([action_emb_dim] * action_head_layers + [action_dim],
                                   activation=nn.GELU,
                                   output_activation=nn.Identity)
        self.state_pred_head = nn.Linear(embedding_dim, state_dim)
        self.cost_pred_head = nn.Linear(embedding_dim, 2)

        # Dynamic heads based on flags
        if self.use_cost:
            self.cost_pred_head = nn.Linear(embedding_dim, 2)
        else:
            self.cost_pred_head = None

        if self.use_rew:
            self.return_pred_head = nn.Linear(embedding_dim, 1)
        else:
            self.return_pred_head = None

        if self.stochastic:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

        self.apply(self._init_weights)

    def temperature(self):
        if self.stochastic:
            return self.log_temperature.exp()
        else:
            return None

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _embed_inputs(self, states, actions, returns_to_go, costs_to_go, time_steps):
        batch_size, seq_len = states.shape[0], states.shape[1]
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        seq_list = [state_emb, act_emb]

        if self.cost_transform is not None and self.use_cost:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        if self.use_cost:
            costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, costs_emb)
        if self.use_rew:
            returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, returns_emb)

        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len, self.embedding_dim)
        return sequence

    def load_pretrained_models(self, return_model_path: str, cost_model_path: str, merge_method: str = "weighted"):
        """
        加载并合并预训练的return模型和cost模型参数

        Args:
            return_model_path: return模型的路径
            cost_model_path: cost模型的路径
            merge_method: 合并方法，可选["weighted", "attention", "gated"]
        """
        return_state = torch.load(return_model_path)["model_state"]
        cost_state = torch.load(cost_model_path)["model_state"]

        merged_state = {}

        if merge_method == "weighted":
            # 使用加权平均合并参数
            for key in self.state_dict().keys():
                if 'causal_mask' in key:
                    continue

                if key in return_state and key in cost_state:
                    # 对共同存在的参数进行加权平均
                    weight_return = 0.5  # 可以根据验证集表现动态调整权重
                    weight_cost = 1 - weight_return
                    merged_state[key] = weight_return * return_state[key] + weight_cost * cost_state[key]
                elif key in return_state:
                    merged_state[key] = return_state[key]
                elif key in cost_state:
                    merged_state[key] = cost_state[key]

        elif merge_method == "attention":
            # 使用注意力机制合并参数
            for key in self.state_dict().keys():
                if 'causal_mask' in key:
                    continue

                if key in return_state and key in cost_state:
                    # 计算注意力权重
                    return_param = return_state[key]
                    cost_param = cost_state[key]

                    # 使用参数的范数作为注意力分数
                    return_norm = torch.norm(return_param)
                    cost_norm = torch.norm(cost_param)

                    attention_weights = torch.softmax(torch.tensor([return_norm, cost_norm]), dim=0)
                    merged_state[key] = attention_weights[0] * return_param + attention_weights[1] * cost_param
                elif key in return_state:
                    merged_state[key] = return_state[key]
                elif key in cost_state:
                    merged_state[key] = cost_state[key]

        elif merge_method == "gated":
            # 使用门控机制合并参数
            for key in self.state_dict().keys():
                if 'causal_mask' in key:
                    continue

                if key in return_state and key in cost_state:
                    return_param = return_state[key]
                    cost_param = cost_state[key]

                    # 创建门控参数
                    gate = torch.sigmoid(torch.abs(return_param - cost_param))
                    merged_state[key] = gate * return_param + (1 - gate) * cost_param
                elif key in return_state:
                    merged_state[key] = return_state[key]
                elif key in cost_state:
                    merged_state[key] = cost_state[key]

        # 加载合并后的参数
        missing_keys, unexpected_keys = self.load_state_dict(merged_state, strict=False)

        print(f"Successfully loaded and merged pretrained models using {merge_method} method")
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor, 
            returns_to_go: torch.Tensor,
            costs_to_go: torch.Tensor,
            time_steps: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            episode_cost: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        标准的前向传播函数
        """
        # 获取return和cost的预测结果
        sequence = self._embed_inputs(states, actions, returns_to_go, costs_to_go, time_steps)
        
        if padding_mask is not None:
            padding_mask = torch.stack([padding_mask] * self.seq_repeat,
                                       dim=1).permute(0, 2, 1).reshape(sequence.shape[0], -1)

        if self.cost_prefix:
            episode_cost = episode_cost.unsqueeze(-1).unsqueeze(-1).to(states.dtype)
            episode_cost_emb = self.prefix_emb(episode_cost)
            sequence = torch.cat([episode_cost_emb, sequence], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        # 添加融合层的处理逻辑
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        # 通过transformer blocks
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)
            
        out = self.out_norm(out)
        
        if self.cost_prefix:
            out = out[:, 1:]
            
        # 动态计算序列长度
        batch_size = states.shape[0]
        actual_seq_len = states.shape[1]
        seq_repeat = out.shape[1] // actual_seq_len

        # 重塑输出
        out = out.reshape(batch_size, actual_seq_len, seq_repeat, self.embedding_dim)
        out = out.permute(0, 2, 1, 3)

        # 提取特征
        action_feature = out[:, seq_repeat - 1]
        state_feat = out[:, seq_repeat - 2]

        # 融合return和cost的预测
        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + self.cost_emb(costs_to_go.unsqueeze(-1)).detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * self.cost_emb(costs_to_go.unsqueeze(-1)).detach()
        if self.cat_cost_feat and self.use_cost:
            state_feat = torch.cat([state_feat, self.cost_emb(costs_to_go.unsqueeze(-1)).detach()], dim=2)

        # 预测动作
        action_preds = self.action_head(state_feat)
        state_preds = self.state_pred_head(action_feature)

        # 预测cost
        cost_preds = None
        if self.use_cost and self.cost_pred_head is not None:
            cost_preds = self.cost_pred_head(action_feature)
            cost_preds = F.log_softmax(cost_preds, dim=-1)

        # 预测return  
        return_preds = None
        if self.use_rew and self.return_pred_head is not None:
            return_preds = self.return_pred_head(action_feature)

        return action_preds, cost_preds, state_preds, return_preds

    def merge_forward(self, *args, **kwargs):
        """
        融合模型的前向传播函数，直接调用标准forward
        """
        return self.forward(*args, **kwargs)

class CDTTrainer:
    """
    Constrained Decision Transformer Trainer
    """

    def __init__(
            self,
            model: CDT,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            betas: Tuple[float, ...] = (0.9, 0.999),
            clip_grad: float = 0.25,
            lr_warmup_steps: int = 10000,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            loss_cost_weight: float = 0.0,
            loss_state_weight: float = 0.0,
            loss_return_weight: float = 1.0,  # New return loss weight
            cost_reverse: bool = False,
            no_entropy: bool = False,
            device="cpu"
    ) -> None:
        self.model = model
        self.logger = logger
        self.env = env
        self.clip_grad = clip_grad
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.cost_weight = loss_cost_weight
        self.state_weight = loss_state_weight
        self.return_weight = loss_return_weight  # New
        self.cost_reverse = cost_reverse
        self.no_entropy = no_entropy

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )
        self.stochastic = self.model.stochastic
        if self.stochastic:
            self.log_temperature_optimizer = torch.optim.Adam(
                [self.model.log_temperature],
                lr=1e-4,
                betas=[0.9, 0.999],
            )
        self.max_action = self.model.max_action

        self.beta_dist = Beta(torch.tensor(2, dtype=torch.float, device=self.device),
                              torch.tensor(5, dtype=torch.float, device=self.device))

    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask,
                       episode_cost, costs):
        padding_mask = ~mask.to(torch.bool)
        action_preds, cost_preds, state_preds, return_preds = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            costs_to_go=costs_return,
            time_steps=time_steps,
            padding_mask=padding_mask,
            episode_cost=episode_cost,
        )

        # Action loss
        if self.stochastic:
            log_likelihood = action_preds.log_prob(actions)[mask > 0].mean()
            entropy = action_preds.entropy()[mask > 0].mean()
            entropy_reg = self.model.temperature().detach()
            entropy_reg_item = entropy_reg.item()
            if self.no_entropy:
                entropy_reg = 0.0
                entropy_reg_item = 0.0
            act_loss = -(log_likelihood + entropy_reg * entropy)
            self.logger.store(tab="train",
                              nll=-log_likelihood.item(),
                              ent=entropy.item(),
                              ent_reg=entropy_reg_item)
        else:
            act_loss = F.mse_loss(action_preds, actions.detach(), reduction="none")
            act_loss = (act_loss * mask.unsqueeze(-1)).mean()

        # Cost loss
        cost_loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if self.model.use_cost and cost_preds is not None:
            cost_preds = cost_preds.reshape(-1, 2)
            costs = costs.flatten().long().detach()
            cost_loss = F.nll_loss(cost_preds, costs, reduction="none")
            cost_loss = (cost_loss * mask.flatten()).mean()
            pred = cost_preds.data.max(dim=1)[1]
            correct = pred.eq(costs.data.view_as(pred)) * mask.flatten()
            acc = correct.sum() / mask.sum()

        # Return loss (new)
        return_loss = torch.tensor(0.0, device=self.device)
        if self.model.use_rew and return_preds is not None:
            return_loss = F.mse_loss(return_preds.squeeze(-1), returns.detach(), reduction="none")
            return_loss = (return_loss * mask).mean()

        # State loss
        state_loss = F.mse_loss(state_preds[:, :-1], states[:, 1:].detach(), reduction="none")
        state_loss = (state_loss * mask[:, :-1].unsqueeze(-1)).mean()

        # Total loss
        loss = act_loss + \
               self.cost_weight * cost_loss + \
               self.state_weight * state_loss + \
               self.return_weight * return_loss

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()

        if self.stochastic:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (self.model.temperature() *
                                (entropy - self.model.target_entropy).detach())
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        self.scheduler.step()
        self.logger.store(
            tab="train",
            all_loss=loss.item(),
            act_loss=act_loss.item(),
            cost_loss=cost_loss.item() if self.model.use_cost else 0.0,
            cost_acc=acc.item() if self.model.use_cost else 0.0,
            return_loss=return_loss.item() if self.model.use_rew else 0.0,
            state_loss=state_loss.item(),
            train_lr=self.scheduler.get_last_lr()[0],
        )

    # ... [保留原有的 evaluate, rollout 等方法]

    def evaluate(self, num_rollouts, target_return, target_cost):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(self.model, self.env,
                                                      target_return, target_cost)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens)

    @torch.no_grad()
    def rollout(
            self,
            model: CDT,
            env: gym.Env,
            target_return: float,
            target_cost: float,
    ) -> Tuple[float, float]:
        """
        Evaluates the performance of the model on a single episode.
        """
        states = torch.zeros(1,
                             model.episode_len + 1,
                             model.state_dim,
                             dtype=torch.float,
                             device=self.device)
        actions = torch.zeros(1,
                              model.episode_len,
                              model.action_dim,
                              dtype=torch.float,
                              device=self.device)
        returns = torch.zeros(1,
                              model.episode_len + 1,
                              dtype=torch.float,
                              device=self.device)
        costs = torch.zeros(1,
                            model.episode_len + 1,
                            dtype=torch.float,
                            device=self.device)
        time_steps = torch.arange(model.episode_len,
                                  dtype=torch.long,
                                  device=self.device)
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)
        returns[:, 0] = torch.as_tensor(target_return, device=self.device)
        costs[:, 0] = torch.as_tensor(target_cost, device=self.device)

        epi_cost = torch.tensor(np.array([target_cost]),
                                dtype=torch.float,
                                device=self.device)

        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for step in range(model.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important) # fix this noqa!!!
            s = states[:, :step + 1][:, -model.seq_len:]  # noqa
            a = actions[:, :step + 1][:, -model.seq_len:]  # noqa
            r = returns[:, :step + 1][:, -model.seq_len:]  # noqa
            c = costs[:, :step + 1][:, -model.seq_len:]  # noqa
            t = time_steps[:, :step + 1][:, -model.seq_len:]  # noqa

            # acts, _, _ = model(s, a, r, c, t, None, epi_cost)
            # 解包修正：捕获所有返回值
            acts, *_ = model(s, a, r, c, t, None, epi_cost)
            if self.stochastic:
                acts = acts.mean
            acts = acts.clamp(-self.max_action, self.max_action)
            act = acts[0, -1].cpu().numpy()
            # act = self.get_ensemble_action(1, model, s, a, r, c, t, epi_cost)

            obs_next, reward, terminated, truncated, info = env.step(act)
            if self.cost_reverse:
                cost = (1.0 - info["cost"]) * self.cost_scale
            else:
                cost = info["cost"] * self.cost_scale
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(act)
            states[:, step + 1] = torch.as_tensor(obs_next)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
            costs[:, step + 1] = torch.as_tensor(costs[:, step] - cost)

            obs = obs_next

            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost


    def get_ensemble_action(self, size: int, model, s, a, r, c, t, epi_cost):
        # [size, seq_len, state_dim]
        s = torch.repeat_interleave(s, size, 0)
        # [size, seq_len, act_dim]
        a = torch.repeat_interleave(a, size, 0)
        # [size, seq_len]
        r = torch.repeat_interleave(r, size, 0)
        c = torch.repeat_interleave(c, size, 0)
        t = torch.repeat_interleave(t, size, 0)
        epi_cost = torch.repeat_interleave(epi_cost, size, 0)

        acts, _, _ = model(s, a, r, c, t, None, epi_cost)
        if self.stochastic:
            acts = acts.mean

        # [size, seq_len, act_dim]
        acts = torch.mean(acts, dim=0, keepdim=True)
        acts = acts.clamp(-self.max_action, self.max_action)
        act = acts[0, -1].cpu().numpy()
        return act

    def collect_random_rollouts(self, num_rollouts):
        episode_rets = []
        for _ in range(num_rollouts):
            obs, info = self.env.reset()
            episode_ret = 0.0
            for step in range(self.model.episode_len):
                act = self.env.action_space.sample()
                obs_next, reward, terminated, truncated, info = self.env.step(act)
                obs = obs_next
                episode_ret += reward
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
        return np.mean(episode_rets)
