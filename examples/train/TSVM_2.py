import torch
import argparse
import math
import copy
from collections import OrderedDict
from typing import List, Optional, Dict, Any, Iterable, Union

# 定义StateDictType类型
StateDictType = Dict[str, torch.Tensor]


# 参数名称匹配检查
def check_parameterNamesMatch(checkpoints):
    """
    检查多个检查点之间的参数名称是否匹配

    参数:
        checkpoints (list): 状态字典列表

    异常:
        ValueError: 如果参数名称不匹配，则抛出异常
    """
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "模型中的参数名称不同。 "
                    f"不同的参数为 {parameter_names.symmetric_difference(current_parameterNames)}"
                )


# 状态字典相等性检查
def check_state_dicts_equal(
        state_dict1: StateDictType, state_dict2: StateDictType
) -> bool:
    """
    检查两个状态字典是否相等

    参数:
        state_dict1 (dict): 第一个状态字典
        state_dict2 (dict): 第二个状态字典

    返回:
        bool: 如果状态字典相等则为True，否则为False
    """
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


# 状态字典转向量
def state_dict_to_vector(state_dict, remove_keys=[]):
    """
    将状态字典转换为向量，移除指定的键

    参数:
        state_dict (dict): 要转换的状态字典
        remove_keys (list): 要从状态字典中移除的键列表

    返回:
        Tensor: 状态字典的向量表示
    """
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


# 向量转状态字典
def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    """
    将向量转换回状态字典，移除指定的键

    参数:
        vector (Tensor): 要转换的向量
        state_dict (dict): 参考状态字典
        remove_keys (list): 要从状态字典中移除的键列表

    返回:
        dict: 向量的状态字典表示
    """
    # 创建一个参考字典来定义向量的顺序
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # 使用参考字典创建共享状态字典
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # 添加回编码器和解码器嵌入权重
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


# 状态字典操作函数
def state_dict_sub(a, b, strict=True, device=None):
    """返回两个状态字典之间的差异 a-b"""
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = OrderedDict()
    for k in a:
        if k in b:
            # 处理嵌套字典的情况
            if isinstance(a[k], OrderedDict) and isinstance(b[k], OrderedDict):
                # 递归处理嵌套字典
                diff[k] = state_dict_sub(a[k], b[k], strict=strict, device=device)
            elif isinstance(a[k], torch.Tensor) and isinstance(b[k], torch.Tensor):
                # 检查张量类型，对布尔张量使用异或操作
                if a[k].dtype == torch.bool and b[k].dtype == torch.bool:
                    # 确保两个张量在同一设备上
                    if a[k].device != b[k].device:
                        # 如果指定了目标设备，则使用它，否则使用a[k]的设备
                        target_device = device if device is not None else a[k].device
                        a_tensor = a[k].to(target_device)
                        b_tensor = b[k].to(target_device)
                        diff[k] = a_tensor ^ b_tensor  # 使用异或操作
                    else:
                        diff[k] = a[k] ^ b[k]  # 使用异或操作
                else:
                    # 标准张量减法
                    # 确保两个张量在同一设备上
                    if a[k].device != b[k].device:
                        # 如果指定了目标设备，则使用它，否则使用a[k]的设备
                        target_device = device if device is not None else a[k].device
                        a_tensor = a[k].to(target_device)
                        b_tensor = b[k].to(target_device)
                        diff[k] = a_tensor - b_tensor
                    else:
                        diff[k] = a[k] - b[k]
                
                # 如果指定了设备，将结果移到该设备上
                if device is not None and diff[k].device != device:
                    diff[k] = diff[k].to(device, non_blocking=True)
            else:
                print(f"警告: 键 {k} 的类型不支持相减操作，跳过。a类型: {type(a[k])}, b类型: {type(b[k])}")
    return diff


def state_dict_add(a, b, strict=True, device=None):
    """返回两个状态字典的和"""
    result = OrderedDict()
    if strict:
        for key in a:
            # 处理嵌套字典的情况
            if isinstance(a[key], OrderedDict) and isinstance(b[key], OrderedDict):
                # 递归处理嵌套字典
                result[key] = state_dict_add(a[key], b[key], strict=strict, device=device)
            elif isinstance(a[key], torch.Tensor) and isinstance(b[key], torch.Tensor):
                # 检查张量类型，对布尔张量使用逻辑或操作
                if a[key].dtype == torch.bool and b[key].dtype == torch.bool:
                    # 确保两个张量在同一设备上
                    if a[key].device != b[key].device:
                        # 如果指定了目标设备，则使用它，否则使用a[key]的设备
                        target_device = device if device is not None else a[key].device
                        a_tensor = a[key].to(target_device)
                        b_tensor = b[key].to(target_device)
                        result[key] = a_tensor | b_tensor
                    else:
                        result[key] = a[key] | b[key]
                else:
                    # 标准张量加法
                    # 确保两个张量在同一设备上
                    if a[key].device != b[key].device:
                        # 如果指定了目标设备，则使用它，否则使用a[key]的设备
                        target_device = device if device is not None else a[key].device
                        a_tensor = a[key].to(target_device)
                        b_tensor = b[key].to(target_device)
                        result[key] = a_tensor + b_tensor
                    else:
                        result[key] = a[key] + b[key]
                
                # 如果指定了设备，将结果移到该设备上
                if device is not None and result[key].device != device:
                    result[key] = result[key].to(device, non_blocking=True)
            else:
                print(f"警告: 键 {key} 的类型不支持相加操作，跳过。a类型: {type(a[key])}, b类型: {type(b[key])}")
    else:
        for key in a:
            if key in b:
                # 处理嵌套字典的情况
                if isinstance(a[key], OrderedDict) and isinstance(b[key], OrderedDict):
                    # 递归处理嵌套字典
                    result[key] = state_dict_add(a[key], b[key], strict=False, device=device)
                elif isinstance(a[key], torch.Tensor) and isinstance(b[key], torch.Tensor):
                    # 检查张量类型，对布尔张量使用逻辑或操作
                    if a[key].dtype == torch.bool and b[key].dtype == torch.bool:
                        # 确保两个张量在同一设备上
                        if a[key].device != b[key].device:
                            # 如果指定了目标设备，则使用它，否则使用a[key]的设备
                            target_device = device if device is not None else a[key].device
                            a_tensor = a[key].to(target_device)
                            b_tensor = b[key].to(target_device)
                            result[key] = a_tensor | b_tensor
                        else:
                            result[key] = a[key] | b[key]
                    else:
                        # 标准张量加法
                        # 确保两个张量在同一设备上
                        if a[key].device != b[key].device:
                            # 如果指定了目标设备，则使用它，否则使用a[key]的设备
                            target_device = device if device is not None else a[key].device
                            a_tensor = a[key].to(target_device)
                            b_tensor = b[key].to(target_device)
                            result[key] = a_tensor + b_tensor
                        else:
                            result[key] = a[key] + b[key]
                    
                    # 如果指定了设备，将结果移到该设备上
                    if device is not None and result[key].device != device:
                        result[key] = result[key].to(device, non_blocking=True)
                else:
                    print(f"警告: 键 {key} 的类型不支持相加操作，跳过。a类型: {type(a[key])}, b类型: {type(b[key])}")
            else:
                # 如果键仅存在于a中，直接复制
                result[key] = a[key]
        
        # 处理仅存在于b中的键
        for key in b:
            if key not in a:
                result[key] = b[key]
    
    return result


def state_dict_mul(state_dict, scalar, device=None):
    """返回状态字典乘以标量的结果"""
    result = OrderedDict()
    for k in state_dict:
        # 处理嵌套字典的情况
        if isinstance(state_dict[k], OrderedDict):
            # 递归处理嵌套字典
            result[k] = state_dict_mul(state_dict[k], scalar, device=device)
        elif isinstance(state_dict[k], torch.Tensor):
            # 对布尔张量的特殊处理
            if state_dict[k].dtype == torch.bool:
                # 如果scalar为0，返回全False张量；否则保持不变
                if scalar == 0:
                    result[k] = torch.zeros_like(state_dict[k], dtype=torch.bool)
                else:
                    result[k] = state_dict[k].clone()
            else:
                # 标准张量乘法
                result[k] = scalar * state_dict[k]
            
            # 如果指定了设备，将结果移到该设备上
            if device is not None and result[k].device != device:
                result[k] = result[k].to(device, non_blocking=True)
        else:
            print(f"警告: 键 {k} 的类型不支持乘法操作，跳过。类型: {type(state_dict[k])}")
    return result


# 1. 标准TSVM实现
@torch.no_grad()
def compute_and_sum_svd_mem_reduction(
        task_vectors: List[Dict[str, torch.Tensor]],
        exclude_keys: Optional[List[str]] = None,
        accelerator: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> Dict[str, torch.Tensor]:
    """
    计算每个任务向量的SVD，根据sv_reduction因子降低向量的维度，
    并连接低秩矩阵。如果向量不是2D张量或在exclude_keys中，计算平均值。
    对第二个操作也执行SVD计算。

    参数:
        task_vectors (list): 任务向量列表，每个向量包含一个字典
        exclude_keys (list): 从TSVM排除的键列表
        accelerator (torch.device): 用于计算的设备

    返回:
        dict: 包含SVD计算和合并后的新向量的字典
    """
    if exclude_keys is None:
        exclude_keys = []
    sv_reduction = 1 / len(task_vectors)

    new_vector = OrderedDict()

    # 递归处理嵌套字典
    def process_nested_dict(task_vecs, current_path=""):
        """递归处理嵌套字典结构"""
        result = OrderedDict()

        # 先获取第一个向量的结构
        first_vec = task_vecs[0]
        for key in first_vec:
            full_key = f"{current_path}.{key}" if current_path else key

            # 如果是嵌套字典，递归处理
            if isinstance(first_vec[key], OrderedDict):
                nested_vecs = [tv[key] for tv in task_vecs]
                result[key] = process_nested_dict(nested_vecs, full_key)
                continue

            # 跳过非张量类型
            if not isinstance(first_vec[key], torch.Tensor):
                print(f"警告: 键 {full_key} 不是张量，跳过。类型: {type(first_vec[key])}")
                continue

            original_device = first_vec[key].device
            original_dtype = first_vec[key].dtype

            for i, task_vector in enumerate(task_vecs):
                vec = task_vector[key].to(accelerator)

                if len(vec.shape) == 2 and full_key not in exclude_keys:
                    # SVD不支持半精度，需要转换为float32
                    if not (original_dtype == torch.float32 or original_dtype == torch.float64):
                        vec = vec.to(dtype=torch.float32)
                    
                    # 特别处理布尔张量
                    if original_dtype == torch.bool:
                        print(f"警告: 键 {full_key} 是布尔类型，将其转换为float32以便SVD计算")

                    try:
                        u, s, v = torch.linalg.svd(vec, full_matrices=False)

                        if i == 0:
                            print(f"计算SVD: {full_key}...")
                            sum_u = torch.zeros_like(u, device=accelerator)
                            sum_s = torch.zeros_like(s, device=accelerator)
                            sum_v = torch.zeros_like(v, device=accelerator)
                        reduced_index_s = int(s.shape[0] * sv_reduction)

                        # 选择前reduced_index_s列的u
                        sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                        sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[:reduced_index_s]
                        # 选择前reduced_index_s行的v
                        sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                    except Exception as e:
                        print(f"SVD计算失败，对键 {full_key} 使用均值: {str(e)}")
                        if i == 0:
                            result[key] = vec.clone()
                        else:
                            result[key] += (vec - result[key]) / (i + 1)
                else:
                    # 如果向量不是2D张量或在exclude_keys中，计算均值
                    if i == 0:
                        result[key] = vec.clone()
                    else:
                        result[key] += (vec - result[key]) / (i + 1)

            if len(first_vec[key].shape) == 2 and full_key not in exclude_keys and 'sum_u' in locals():
                try:
                    u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                    u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                    result[key] = torch.linalg.multi_dot(
                        (
                            u_u,
                            v_u,
                            torch.diag(sum_s),
                            u_v,
                            v_v,
                        )
                    )
                except Exception as e:
                    print(f"第二次SVD计算失败，对键 {full_key} 使用当前结果: {str(e)}")
                    # 如果第二次SVD失败，保留当前结果

            # 确保键在result中
            if key not in result:
                continue

            result[key] = result[key].to(
                device=original_device, dtype=original_dtype, non_blocking=True
            )

        return result

    # 开始处理
    return process_nested_dict(task_vectors)


# 2. 无损TSVM实现
@torch.no_grad()
def compute_and_sum_svd_mem_reduction_lossless(
        task_vectors: List[Dict[str, torch.Tensor]],
        accelerator: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    """无损版TSVM实现"""
    print("执行无损SVD计算...")
    
    # 递归处理嵌套字典
    def process_nested_dict(task_vecs, current_path=""):
        """递归处理嵌套字典结构"""
        result = OrderedDict()

        # 先获取第一个向量的结构
        first_vec = task_vecs[0]
        for key in first_vec:
            full_key = f"{current_path}.{key}" if current_path else key

            # 如果是嵌套字典，递归处理
            if isinstance(first_vec[key], OrderedDict):
                nested_vecs = [tv[key] for tv in task_vecs]
                result[key] = process_nested_dict(nested_vecs, full_key)
                continue

            # 跳过非张量类型
            if not isinstance(first_vec[key], torch.Tensor):
                print(f"警告: 键 {full_key} 不是张量，跳过。类型: {type(first_vec[key])}")
                continue

            original_device = first_vec[key].device
            
            for i, task_vector in enumerate(task_vecs):
                vec = task_vector[key].to(accelerator)

                if len(vec.shape) == 2 and "text_projection" not in full_key:
                    # 检查张量类型，如果是布尔类型，则转换为float
                    if vec.dtype == torch.bool:
                        print(f"警告: 键 {full_key} 是布尔类型，将其转换为float32以便SVD计算")
                        vec = vec.to(dtype=torch.float32)
                    
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"计算SVD: {full_key}...")
                        sum_u = torch.zeros(
                            u.shape[0],
                            u.shape[1] * len(task_vecs),
                            device=accelerator,
                        )
                        sum_s = torch.zeros(
                            s.shape[0] * len(task_vecs), device=accelerator
                        )
                        sum_v = torch.zeros(
                            v.shape[0] * len(task_vecs),
                            v.shape[1],
                            device=accelerator,
                        )
                    reduced_index_s = s.shape[0]

                    # 将u的前reduced_index_s列放置
                    sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    # 将v的前reduced_index_s行放置
                    sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

                else:
                    if i == 0:
                        result[key] = vec.clone()
                    else:
                        result[key] += (vec - result[key]) / (i + 1)

            if len(first_vec[key].shape) == 2 and "text_projection" not in full_key and 'sum_u' in locals():
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                result[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )
                
            # 确保键在result中
            if key in result:
                result[key] = result[key].to(original_device, non_blocking=True)

        return result

    # 开始处理
    return process_nested_dict(task_vectors)


# 3. 使用特征分解的无损TSVM实现
@torch.no_grad()
def compute_and_sum_svd_mem_reduction_lossless_eigen(
        task_vectors: List[Dict[str, torch.Tensor]],
        accelerator: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    """使用特征分解的无损版TSVM实现"""
    print("执行无损特征分解SVD计算...")
    
    # 递归处理嵌套字典
    def process_nested_dict(task_vecs, current_path=""):
        """递归处理嵌套字典结构"""
        result = OrderedDict()

        # 先获取第一个向量的结构
        first_vec = task_vecs[0]
        for key in first_vec:
            full_key = f"{current_path}.{key}" if current_path else key

            # 如果是嵌套字典，递归处理
            if isinstance(first_vec[key], OrderedDict):
                nested_vecs = [tv[key] for tv in task_vecs]
                result[key] = process_nested_dict(nested_vecs, full_key)
                continue

            # 跳过非张量类型
            if not isinstance(first_vec[key], torch.Tensor):
                print(f"警告: 键 {full_key} 不是张量，跳过。类型: {type(first_vec[key])}")
                continue

            original_device = first_vec[key].device
            
            for i, task_vector in enumerate(task_vecs):
                vec = task_vector[key].to(accelerator)

                if len(vec.shape) == 2 and "text_projection" not in full_key:
                    # 检查张量类型，如果是布尔类型，则转换为float
                    if vec.dtype == torch.bool:
                        print(f"警告: 键 {full_key} 是布尔类型，将其转换为float32以便SVD计算")
                        vec = vec.to(dtype=torch.float32)
                    
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"计算SVD: {full_key}...")
                        sum_u = torch.zeros(
                            u.shape[0],
                            u.shape[1] * len(task_vecs),
                            device=accelerator,
                        )
                        sum_s = torch.zeros(
                            s.shape[0] * len(task_vecs), device=accelerator
                        )
                        sum_v = torch.zeros(
                            v.shape[0] * len(task_vecs),
                            v.shape[1],
                            device=accelerator,
                        )
                    reduced_index_s = s.shape[0]

                    # 将u的前reduced_index_s列放置
                    sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    # 将v的前reduced_index_s行放置
                    sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

                else:
                    if i == 0:
                        result[key] = vec.clone()
                    else:
                        result[key] += (vec - result[key]) / (i + 1)

            if len(first_vec[key].shape) == 2 and "text_projection" not in full_key and 'sum_u' in locals():
                sum_s, indices = torch.sort(sum_s, stable=True)

                sum_u = torch.index_select(sum_u, 1, indices)
                l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
                u_orth = (
                        q_u
                        @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
                        @ q_u.mT
                )

                sum_v = torch.index_select(sum_v, 0, indices)

                l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
                v_orth = (
                        q_v
                        @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
                        @ q_v.mT
                )

                result[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        u_orth,
                        torch.diag(sum_s),
                        v_orth,
                        sum_v,
                    )
                )
                
            # 确保键在result中
            if key in result:
                result[key] = result[key].to(original_device, non_blocking=True)

        return result

    # 开始处理
    return process_nested_dict(task_vectors)


# 4. 使用特征分解的TSVM实现
@torch.no_grad()
def compute_and_sum_svd_mem_reduction_2(
        task_vectors: List[Dict[str, torch.Tensor]],
        accelerator: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    """使用特征分解的TSVM实现"""
    sv_reduction = 1 / len(task_vectors)
    print("执行特征分解SVD计算...")

    # 递归处理嵌套字典
    def process_nested_dict(task_vecs, current_path=""):
        """递归处理嵌套字典结构"""
        result = OrderedDict()

        # 先获取第一个向量的结构
        first_vec = task_vecs[0]
        for key in first_vec:
            full_key = f"{current_path}.{key}" if current_path else key

            # 如果是嵌套字典，递归处理
            if isinstance(first_vec[key], OrderedDict):
                nested_vecs = [tv[key] for tv in task_vecs]
                result[key] = process_nested_dict(nested_vecs, full_key)
                continue

            # 跳过非张量类型
            if not isinstance(first_vec[key], torch.Tensor):
                print(f"警告: 键 {full_key} 不是张量，跳过。类型: {type(first_vec[key])}")
                continue

            original_device = first_vec[key].device
            
            for i, task_vector in enumerate(task_vecs):
                vec = task_vector[key].to(accelerator)

                if len(vec.shape) == 2 and "text_projection" not in full_key:
                    # 检查张量类型，如果是布尔类型，则转换为float
                    if vec.dtype == torch.bool:
                        print(f"警告: 键 {full_key} 是布尔类型，将其转换为float32以便SVD计算")
                        vec = vec.to(dtype=torch.float32)
                    
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"计算SVD: {full_key}...")
                        sum_u = torch.zeros_like(u, device=accelerator)
                        sum_s = torch.zeros_like(s, device=accelerator)
                        sum_v = torch.zeros_like(v, device=accelerator)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

                else:
                    if i == 0:
                        result[key] = vec.clone()
                    else:
                        result[key] += (vec - result[key]) / (i + 1)

            if len(first_vec[key].shape) == 2 and "text_projection" not in full_key and 'sum_u' in locals():
                sum_s, indices = torch.sort(sum_s, stable=True)

                sum_u = torch.index_select(sum_u, 1, indices)
                l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
                u_orth = (
                        q_u
                        @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
                        @ q_u.mT
                )

                sum_v = torch.index_select(sum_v, 0, indices)

                l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
                v_orth = (
                        q_v
                        @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
                        @ q_v.mT
                )

                result[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        u_orth,
                        torch.diag(sum_s),
                        v_orth,
                        sum_v,
                    )
                )
                
            # 确保键在result中
            if key in result:
                result[key] = result[key].to(original_device, non_blocking=True)

        return result

    # 开始处理
    return process_nested_dict(task_vectors)


# 5. 秩降低版TSVM实现
@torch.no_grad()
def compute_and_sum_svd_mem_reduction_rank_reduction(
        task_vectors: List[Dict[str, torch.Tensor]],
        accelerator: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    """秩降低版TSVM实现"""
    sv_reduction = 1 / len(task_vectors)
    print("执行秩降低SVD计算...")

    # 递归处理嵌套字典
    def process_nested_dict(task_vecs, current_path=""):
        """递归处理嵌套字典结构"""
        result = OrderedDict()

        # 先获取第一个向量的结构
        first_vec = task_vecs[0]
        for key in first_vec:
            full_key = f"{current_path}.{key}" if current_path else key

            # 如果是嵌套字典，递归处理
            if isinstance(first_vec[key], OrderedDict):
                nested_vecs = [tv[key] for tv in task_vecs]
                result[key] = process_nested_dict(nested_vecs, full_key)
                continue

            # 跳过非张量类型
            if not isinstance(first_vec[key], torch.Tensor):
                print(f"警告: 键 {full_key} 不是张量，跳过。类型: {type(first_vec[key])}")
                continue

            original_device = first_vec[key].device
            original_dtype = first_vec[key].dtype
            
            for i, task_vector in enumerate(task_vecs):
                vec = task_vector[key].to(accelerator)

                if len(vec.shape) == 2 and "text_projection" not in full_key:
                    # 检查张量类型，如果是布尔类型，则转换为float
                    if vec.dtype == torch.bool:
                        print(f"警告: 键 {full_key} 是布尔类型，将其转换为float32以便SVD计算")
                        vec = vec.to(dtype=torch.float32)
                    
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"计算SVD: {full_key}...")
                        sum_u = torch.zeros_like(u, device=accelerator)
                        sum_s = torch.zeros_like(s, device=accelerator)
                        sum_v = torch.zeros_like(v, device=accelerator)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

                else:
                    if i == 0:
                        result[key] = vec.clone()
                    else:
                        result[key] += (vec - result[key]) / (i + 1)

            if len(first_vec[key].shape) == 2 and "text_projection" not in full_key and 'sum_u' in locals():
                result[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v,
                    )
                )
                
            # 确保键在result中
            if key in result:
                result[key] = result[key].to(device=original_device, dtype=original_dtype, non_blocking=True)

        return result

    # 开始处理
    return process_nested_dict(task_vectors)


# 6. 虚拟(dummy)TSVM实现
@torch.no_grad()
def compute_and_sum_svd_mem_reduction_dummy(
        task_vectors: List[Dict[str, torch.Tensor]],
        accelerator: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    """执行虚拟TSVM操作，只对第一个任务向量进行真实SVD分解，其余使用随机正交向量"""
    sv_reduction = 1 / len(task_vectors)
    print("执行虚拟(dummy)SVD计算...")

    # 递归处理嵌套字典
    def process_nested_dict(task_vecs, current_path=""):
        """递归处理嵌套字典结构"""
        result = OrderedDict()

        # 先获取第一个向量的结构
        first_vec = task_vecs[0]
        for key in first_vec:
            full_key = f"{current_path}.{key}" if current_path else key

            # 如果是嵌套字典，递归处理
            if isinstance(first_vec[key], OrderedDict):
                nested_vecs = [tv[key] for tv in task_vecs]
                result[key] = process_nested_dict(nested_vecs, full_key)
                continue

            # 跳过非张量类型
            if not isinstance(first_vec[key], torch.Tensor):
                print(f"警告: 键 {full_key} 不是张量，跳过。类型: {type(first_vec[key])}")
                continue

            original_device = first_vec[key].device
            original_dtype = first_vec[key].dtype
            
            for i, task_vector in enumerate(task_vecs):
                vec = task_vector[key].to(accelerator)

                if len(vec.shape) == 2 and "text_projection" not in full_key:
                    if i == 0:
                        # 检查张量类型，如果是布尔类型，则转换为float
                        if vec.dtype == torch.bool:
                            print(f"警告: 键 {full_key} 是布尔类型，将其转换为float32以便SVD计算")
                            vec = vec.to(dtype=torch.float32)
                        
                        u, s, v = torch.linalg.svd(vec, full_matrices=False)
                        reduced_index_s = int(s.shape[0] * sv_reduction)

                        print(f"计算SVD: {full_key}...")
                        sum_u = torch.zeros_like(u, device=accelerator)
                        sum_s = torch.zeros_like(s, device=accelerator)
                        sum_v = torch.zeros_like(v, device=accelerator)

                        # 选择前reduced_index_s列的u
                        sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                        sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[:reduced_index_s]
                        # 选择前reduced_index_s行的v
                        sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                    else:
                        # 生成与前面向量正交的随机u和v向量
                        print(f"生成随机正交向量代替实际SVD: {full_key}")
                        u = torch.nn.functional.normalize(
                            torch.randn_like(sum_u), p=2, dim=-2
                        )
                        v = torch.nn.functional.normalize(
                            torch.randn_like(sum_v), p=2, dim=-1
                        )

                        # 选择前reduced_index_s列的u
                        sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                        sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[:reduced_index_s]
                        # 选择前reduced_index_s行的v
                        sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

                else:
                    if i == 0:
                        result[key] = vec.clone()
                    else:
                        result[key] += (vec - result[key]) / (i + 1)

            if len(first_vec[key].shape) == 2 and "text_projection" not in full_key and 'sum_u' in locals():
                result[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v,
                    )
                )
                
            # 确保键在result中
            if key in result:
                result[key] = result[key].to(device=original_device, dtype=original_dtype, non_blocking=True)

        return result

    # 开始处理
    return process_nested_dict(task_vectors)


# 7. SVD字典计算
@torch.no_grad()
def compute_svd_dict(
        task_vectors: List[Dict[str, torch.Tensor]],
        dataset_names: List[str] = None,
        accelerator: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    """
    计算每个任务向量的SVD并将结果存储在字典中

    参数:
        task_vectors: 任务向量列表
        dataset_names: 数据集名称列表，如果为None则自动生成
        accelerator: 计算设备

    返回:
        dict: 包含每个数据集SVD组件的字典
    """
    # 如果没有提供数据集名称，则自动生成
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(task_vectors))]

    sv_reduction = 1 / len(task_vectors)
    svd_dict = {}

    # 递归处理嵌套字典
    def process_nested_dict(task_vecs, datasets, current_path=""):
        """递归处理嵌套字典结构"""
        result = {}
        
        # 为每个数据集创建对应的条目
        for dataset in datasets:
            if dataset not in result:
                result[dataset] = {}
        
        # 先获取第一个向量的结构作为参考
        first_vec = task_vecs[0]
        
        for key in first_vec:
            full_key = f"{current_path}.{key}" if current_path else key
            
            # 对于每个数据集处理该键
            for i, (task_vector, dataset) in enumerate(zip(task_vecs, datasets)):
                if key not in result[dataset]:
                    result[dataset][key] = {}
                
                # 如果是嵌套字典，递归处理
                if isinstance(task_vector[key], OrderedDict):
                    # 收集所有任务向量的对应嵌套字典
                    nested_vecs = [tv[key] for tv in task_vecs]
                    # 递归处理并存储结果
                    nested_result = process_nested_dict(nested_vecs, datasets, full_key)
                    for ds in datasets:
                        result[ds][key] = nested_result[ds]
                    continue
                
                # 跳过非张量类型
                if not isinstance(task_vector[key], torch.Tensor):
                    print(f"警告: 键 {full_key} 在数据集 {dataset} 中不是张量，跳过。类型: {type(task_vector[key])}")
                    continue
                
                vec = task_vector[key].to(accelerator)
                
                if len(vec.shape) == 2 and "text_projection" not in full_key:
                    # 检查张量类型，如果是布尔类型，则转换为float
                    if vec.dtype == torch.bool:
                        print(f"警告: 键 {full_key} 在数据集 {dataset} 中是布尔类型，将其转换为float32以便SVD计算")
                        vec = vec.to(dtype=torch.float32)
                    
                    print(f"计算数据集 {dataset} 的SVD: {full_key}...")
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    reduced_index_s = int(s.shape[0] * sv_reduction)
                    
                    temp_u = torch.zeros_like(u, device=accelerator)
                    # 选择前reduced_index_s列的u
                    temp_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    result[dataset][key]["u"] = temp_u
                    
                    temp_s = torch.zeros_like(s, device=accelerator)
                    temp_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    result[dataset][key]["s"] = temp_s
                    
                    # 选择前reduced_index_s行的v
                    temp_v = torch.zeros_like(v, device=accelerator)
                    temp_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                    result[dataset][key]["v"] = temp_v
                else:
                    result[dataset][key]["dim1"] = vec
        
        return result
    
    # 开始处理
    return process_nested_dict(task_vectors, dataset_names)


# 8. 合并SVD字典
@torch.no_grad()
def sum_svd_dict(
        svd_dict: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        dataset_names: List[str] = None,
):
    """
    合并多个数据集的SVD组件并计算新向量

    参数:
        svd_dict: 包含每个数据集SVD组件的字典
        dataset_names: 数据集名称列表，如果为None则使用svd_dict的键

    返回:
        dict: 包含合并后的SVD组件或平均的"dim1"值的字典
    """
    # 如果没有提供数据集名称，则使用svd_dict的键
    if dataset_names is None:
        dataset_names = list(svd_dict.keys())

    print("合并SVD字典...")
    
    # 递归处理嵌套字典
    def process_nested_dict(svd_datasets, datasets, current_path=""):
        """递归处理嵌套字典结构"""
        result = OrderedDict()
        
        # 获取第一个数据集的键作为参考
        first_dataset = datasets[0]
        first_dataset_data = svd_datasets[first_dataset]
        
        for key in first_dataset_data:
            full_key = f"{current_path}.{key}" if current_path else key
            
            # 检查是否有嵌套字典
            if isinstance(first_dataset_data[key], dict) and all(isinstance(svd_datasets[ds][key], dict) for ds in datasets) and \
               not any(k in first_dataset_data[key] for k in ["u", "s", "v", "dim1"]):
                # 递归处理嵌套字典
                nested_datasets = {ds: svd_datasets[ds][key] for ds in datasets}
                result[key] = process_nested_dict(nested_datasets, datasets, full_key)
                continue
                
            # 处理SVD组件或dim1
            if "u" in first_dataset_data[key]:
                print(f"合并键 {full_key} 的SVD组件...")
                # 对SVD分解的组件求和
                sum_u = sum([svd_datasets[dataset][key]["u"] for dataset in datasets])
                sum_s = sum([svd_datasets[dataset][key]["s"] for dataset in datasets])
                sum_v = sum([svd_datasets[dataset][key]["v"] for dataset in datasets])

                # 再次进行SVD分解以确保正交性
                try:
                    u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                    u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                    # 使用多点点积计算最终向量
                    result[key] = torch.linalg.multi_dot(
                        (
                            u_u,
                            v_u,
                            torch.diag(sum_s),
                            u_v,
                            v_v,
                        )
                    )
                except Exception as e:
                    print(f"SVD合并失败，对键 {full_key} 使用平均值: {str(e)}")
                    # 如果SVD失败，改用平均值
                    for i, dataset in enumerate(datasets, start=1):
                        if i == 1:
                            # 创建一个用于平均的张量
                            vec_shape = svd_datasets[dataset][key]["u"].shape[0], svd_datasets[dataset][key]["v"].shape[1]
                            result[key] = torch.zeros(vec_shape, device=svd_datasets[dataset][key]["u"].device)
                            
            elif "dim1" in first_dataset_data[key]:
                # 对于非2D张量，计算均值
                for i, dataset in enumerate(datasets, start=1):
                    if i == 1:
                        result[key] = svd_datasets[dataset][key]["dim1"].clone()
                    else:
                        result[key] += (
                                svd_datasets[dataset][key]["dim1"] - result[key]
                        ) / i
        
        return result
    
    # 开始处理
    return process_nested_dict(svd_dict, dataset_names)


# 辅助函数 - 允许使用SVD字典方法
def merge_with_svd_dict(
        task_vectors: List[Dict[str, torch.Tensor]],
        dataset_names: List[str] = None,
        accelerator: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    """使用SVD字典方法合并任务向量"""
    # 计算SVD字典
    svd_dict = compute_svd_dict(task_vectors, dataset_names, accelerator)

    # 合并SVD字典
    return sum_svd_dict(svd_dict, dataset_names)


# 主函数
def main():
    parser = argparse.ArgumentParser(description="合并预训练模型与两个微调模型")
    parser.add_argument("--pretrained", required=True, help="预训练模型路径")
    parser.add_argument("--return_model", required=True, help="第一个微调模型(return)路径")
    parser.add_argument("--cost_model", required=True, help="第二个微调模型(cost)路径")
    parser.add_argument("--output", default="model.pt", help="输出模型路径")
    parser.add_argument("--alpha", type=float, default=1.0, help="合并系数")
    parser.add_argument("--method", type=str, default="standard",
                        choices=["standard", "lossless", "lossless_eigen", "eigen",
                                 "rank_reduction", "dummy", "svd_dict"],
                        help="TSVM方法选择")
    parser.add_argument("--exclude_keys", nargs='+', default=[], help="从TSVM中排除的键")
    parser.add_argument("--dataset_names", nargs='+', default=None,
                        help="数据集名称(用于svd_dict方法)")
    parser.add_argument("--print_params", action="store_true", help="是否打印模型参数")
    args = parser.parse_args()

    # 加载模型
    print(f"加载预训练模型: {args.pretrained}")
    pretrained = torch.load(args.pretrained, map_location='cpu')

    print(f"加载return模型: {args.return_model}")
    return_model = torch.load(args.return_model, map_location='cpu')

    print(f"加载cost模型: {args.cost_model}")
    cost_model = torch.load(args.cost_model, map_location='cpu')

    # 准备状态字典
    if isinstance(pretrained, dict):
        pretrained_dict = pretrained
    else:
        pretrained_dict = pretrained.state_dict()

    if isinstance(return_model, dict):
        return_dict = return_model
    else:
        return_dict = return_model.state_dict()

    if isinstance(cost_model, dict):
        cost_dict = cost_model
    else:
        cost_dict = cost_model.state_dict()

    # 打印模型参数
    if args.print_params or True:  # 默认打印参数
        print("\n" + "=" * 50)
        print("模型参数信息:")
        print("=" * 50)

        # 打印预训练模型参数
        print("\n预训练模型参数:")
        print("-" * 30)
        print_model_params(pretrained_dict)

        # 打印return模型参数
        print("\nReturn模型参数:")
        print("-" * 30)
        print_model_params(return_dict)

        # 打印cost模型参数
        print("\nCost模型参数:")
        print("-" * 30)
        print_model_params(cost_dict)

        print("=" * 50 + "\n")

    # # 检查键是否匹配
    # if set(pretrained_dict.keys()) != set(return_dict.keys()) or set(pretrained_dict.keys()) != set(cost_dict.keys()):
    #     print("警告: 模型键不完全匹配。只处理共同的键。")
    #
    # # 计算任务向量
    # print("计算任务向量...")
    return_task_vector = state_dict_sub(return_dict, pretrained_dict, strict=False)
    cost_task_vector = state_dict_sub(cost_dict, pretrained_dict, strict=False)

    # 选择TSVM方法并执行
    print(f"使用{args.method}方法执行TSVM合并...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_vectors = [return_task_vector, cost_task_vector]

    if args.method == "standard":
        merged_task_vector = compute_and_sum_svd_mem_reduction(
            task_vectors,
            exclude_keys=args.exclude_keys,
            accelerator=device
        )
    elif args.method == "lossless":
        merged_task_vector = compute_and_sum_svd_mem_reduction_lossless(
            task_vectors,
            accelerator=device
        )
    elif args.method == "lossless_eigen":
        merged_task_vector = compute_and_sum_svd_mem_reduction_lossless_eigen(
            task_vectors,
            accelerator=device
        )
    elif args.method == "eigen":
        merged_task_vector = compute_and_sum_svd_mem_reduction_2(
            task_vectors,
            accelerator=device
        )
    elif args.method == "rank_reduction":
        merged_task_vector = compute_and_sum_svd_mem_reduction_rank_reduction(
            task_vectors,
            accelerator=device
        )
    elif args.method == "dummy":
        merged_task_vector = compute_and_sum_svd_mem_reduction_dummy(
            task_vectors,
            accelerator=device
        )
    elif args.method == "svd_dict":
        # 使用SVD字典方法
        dataset_names = args.dataset_names
        if dataset_names is None:
            dataset_names = ["return_model", "cost_model"]
        merged_task_vector = merge_with_svd_dict(
            task_vectors,
            dataset_names=dataset_names,
            accelerator=device
        )

    # 确保设备一致性
    device = torch.device('cpu')  # 在CPU上执行最终合并以确保兼容性
    
    # 应用alpha
    if args.alpha != 1.0:
        print(f"应用alpha系数: {args.alpha} (目标设备: {device})...")
        merged_task_vector = state_dict_mul(merged_task_vector, args.alpha, device=device)

    print(f"将合并后的任务向量加到预训练模型上 (目标设备: {device})...")
    
    # 将合并后的任务向量加到预训练模型上
    merged_model = state_dict_add(merged_task_vector, pretrained_dict, device=device)

    # 处理输出路径
    output_path = args.output
    # 如果提供的是目录而不是文件名，则添加默认文件名
    if not output_path.endswith(".pt") and not output_path.endswith(".pth"):
        import os
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "model.pt")

    # 保存结果
    print(f"保存合并模型到: {output_path}")
    torch.save(merged_model, output_path)

    # 如果需要打印合并后的模型参数
    if args.print_params:
        print("\n" + "=" * 50)
        print("合并后模型参数:")
        print("=" * 50)
        print_model_params(merged_model)
        print("=" * 50)

    print("完成!")


# 定义打印模型参数的函数
def print_model_params(model_dict, prefix="", depth=0):
    """
    递归打印模型参数信息，处理嵌套字典结构

    参数:
        model_dict (dict): 模型状态字典
        prefix (str): 键的前缀，用于嵌套字典
        depth (int): 当前递归深度
    """
    if depth == 0:
        total_params = 0
        total_tensors = 0
        # 打印模型结构和参数
        print(f"参数总数: {len(model_dict)}")

        # 收集所有张量以计算总参数量
        def count_params(d, prefix=""):
            nonlocal total_params, total_tensors
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, OrderedDict):
                    count_params(v, full_key)
                elif isinstance(v, torch.Tensor):
                    total_tensors += 1
                    total_params += v.numel()

        count_params(model_dict)

        # 按字母顺序排序键并打印
        printed_items = 0

        def print_dict_items(d, prefix="", max_items=10):
            nonlocal printed_items
            for k, v in sorted(d.items()):
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, OrderedDict):
                    print(f"{full_key}: <OrderedDict with {len(v)} items>")
                    if printed_items < max_items:
                        print_dict_items(v, full_key, max_items)
                elif isinstance(v, torch.Tensor):
                    if printed_items < max_items or printed_items >= len(model_dict) - max_items:
                        param_shape = tuple(v.shape)
                        param_size = v.numel() * v.element_size() / (1024 * 1024)  # MB
                        param_type = v.dtype
                        param_device = v.device
                        print(
                            f"{full_key}: 形状{param_shape}, 大小{param_size:.2f}MB, 类型{param_type}, 设备{param_device}")
                    elif printed_items == max_items:
                        print("... [省略中间参数] ...")
                    printed_items += 1
                else:
                    print(f"{full_key}: <{type(v).__name__}>")

        print_dict_items(model_dict)

        # 打印总参数数量
        print(f"参数总量: {total_params:,} ({total_params / (1000 * 1000):.2f}M)")
        print(f"张量总数: {total_tensors}")

        # 统计不同类型的参数
        param_types = {}

        def collect_types(d):
            for v in d.values():
                if isinstance(v, OrderedDict):
                    collect_types(v)
                elif isinstance(v, torch.Tensor):
                    dtype = str(v.dtype)
                    if dtype in param_types:
                        param_types[dtype] += 1
                    else:
                        param_types[dtype] = 1

        collect_types(model_dict)

        print("参数类型分布:")
        for dtype, count in param_types.items():
            print(f"  {dtype}: {count} 个参数")
    else:
        # 打印嵌套字典内容
        for k, v in sorted(model_dict.items()):
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, OrderedDict):
                print("  " * depth + f"{k}: <OrderedDict>")
                print_model_params(v, full_key, depth + 1)
            elif isinstance(v, torch.Tensor):
                print("  " * depth + f"{k}: {tuple(v.shape)}, {v.dtype}")
            else:
                print("  " * depth + f"{k}: {type(v).__name__}")


if __name__ == "__main__":
    main()