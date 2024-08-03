# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types
import torch
import numpy as np
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import required_torch_version
from deepspeed import comm as dist


class MVLion(torch.optim.Optimizer):
    """Implements the 1-bit Adam algorithm. Currently GPU-only.
    For usage example please see https://www.deepspeed.ai/tutorials/onebit-adam/
    For technical details please read https://arxiv.org/abs/2102.02888

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        freeze_step (int, optional): Number of steps for warmup (uncompressed)
            stage before we start using compressed communication. (default 100000)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in 1-bit Adam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
        cuda_aware (boolean, required): Set True if the underlying MPI implementation
            supports CUDA-Aware communication. (default: False)
        comm_backend_name (string, optional): Set to 'mpi' if needed. (default: 'nccl')
    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 params,
                 deepspeed=None,
                 lr=1e-4,
                 freeze_step=10,
                 betas=(0.9, 0.99),
                 weight_decay=0.,
                 cuda_aware=False,
                 comm_backend_name='nccl'):

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super(MVLion, self).__init__(params, defaults)
        self.comm_time = 0.0
        self.step_time = 0.0
        self.ave_step = 1
        self.bk_time = 0.0

        self.deepspeed = deepspeed
        self.adam_freeze_key = False
        self.initialize = False
        self.freeze_step = freeze_step
        self.cuda_aware = cuda_aware
        self.using_pipeline = False

        self.comm_backend_name = comm_backend_name

        assert dist.is_initialized(), "Please initialize the torch distributed backend."
        # Empty initializer. Set handle based on the comm backend as follows.
        self.comm_backend_handle = None
        if self.comm_backend_name == 'nccl':
            assert (
                required_torch_version(min_version=1.8)
            ), "Please use torch 1.8 or greater to enable NCCL backend in 1-bit Adam. Alternatively, please specify 'mpi' as the 'comm_backend_name' in config file to proceed with the MPI backend"
            from deepspeed.runtime.comm.nccl import NcclBackend
            self.using_pipeline = hasattr(self.deepspeed, 'pipeline_enable_backward_allreduce')
            self.comm_backend_handle = NcclBackend(self.deepspeed.mpu)
        elif self.comm_backend_name == 'mpi':
            from deepspeed.runtime.comm.mpi import MpiBackend
            self.comm_backend_handle = MpiBackend(cuda_aware)
        elif self.comm_backend_name == 'hccl':
            from deepspeed.runtime.comm.hccl import HcclBackend
            self.using_pipeline = hasattr(self.deepspeed, 'pipeline_enable_backward_allreduce')
            self.comm_backend_handle = HcclBackend(self.deepspeed.mpu)
        self.size = self.comm_backend_handle.size

        self.divider = int(self.size * 8 / np.gcd(self.size, 8))

    def step(self, closure=None, grads=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()
    
        if grads is None:
            grads_group = [None] * len(self.param_groups)
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads
    
        all_grads = []
        all_updates = []
    
        for group, grads_this_group in zip(self.param_groups, grads_group):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])
    
            for p, grad in zip(group['params'], grads_this_group):
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('1-bit Lion does not support sparse gradients')
    
                state = self.state[p]
    
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
    
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                state['step'] += 1
    
                if 'non_freeze' in group.keys() and group['non_freeze'] is True:
                    all_grads.append(grad)
                else:
                    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1)
                    all_updates.append(update)
                    if 'exp_avg_mask' in group:
                        if update.device != group['exp_avg_mask'].device:
                            group['exp_avg_mask'] = group['exp_avg_mask'].to(device=update.device)
                        update.mul_(group['exp_avg_mask'])
                    exp_avg.mul_(beta2).add_(1 - beta2, grad)
    
        if all_grads:
            grads_tensor = torch.cat([g.view(-1) for g in all_grads])
            dist.all_reduce(grads_tensor)
            grads_tensor.mul_(1 / dist.get_world_size())
            offset = 0
            for grad in all_grads:
                grad.copy_(grads_tensor[offset:offset + grad.numel()].view_as(grad))
                offset += grad.numel()

        if all_updates:
            updates_tensor = torch.cat([u.view(-1) for u in all_updates])
            updates_tensor = binary_quantize_allreduce(updates_tensor)
            offset = 0
            for update in all_updates:
                update.copy_(updates_tensor[offset:offset + update.numel()].view_as(update))
                offset += update.numel()
    
        for group, grads_this_group in zip(self.param_groups, grads_group):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])
    
            for p, grad in zip(group['params'], grads_this_group):
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
    
                state = self.state[p]
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                update = exp_avg.sign()
    
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data
                with torch.no_grad():
                    p.add_(-group['lr'] * update)
    
        return loss

    def load_state_dict(self, state_dict):
        """
        Overrides load_state_dict() to add special handling when loading checkpoints
        """
        # Because at different stage exp_avg_mask may change (e.g.,
        # BERT pre-training seqlen 128 and 512 ), we don't use the exp_avg_mask
        # in checkpoints but always use the one user provided in training script.
        # (See example in DeepSpeedExamples/bing_bert/deepspeed_train.py.)
        # Thus here we keep the exp_avg_mask unchanged when loading checkpoint
        for i, group in enumerate(self.param_groups):
            if 'exp_avg_mask' in group:
                state_dict['param_groups'][i]['exp_avg_mask'] = group['exp_avg_mask']
            elif 'exp_avg_mask' not in group and 'exp_avg_mask' in state_dict['param_groups'][i]:
                state_dict['param_groups'][i].pop('exp_avg_mask')
        super().load_state_dict(state_dict)
        if self.state[self.param_groups[0]['params'][0]]['step'] < self.freeze_step:
            if dist.get_rank() == 0:
                print("Checkpoint loaded and OnebitAdam warmup stage starts/continues.")
            if self.adam_freeze_key is True:
                self.adam_freeze_key = False
                if self.using_pipeline:
                    self.deepspeed.pipeline_enable_backward_allreduce = True
                else:
                    self.deepspeed.enable_backward_allreduce = True
        else:
            if dist.get_rank() == 0:
                print("Checkpoint loaded and OnebitAdam compression stage starts/continues.")
            if self.adam_freeze_key is False:
                self.adam_freeze_key = True
                if self.using_pipeline:
                    self.deepspeed.pipeline_enable_backward_allreduce = False
                else:
                    self.deepspeed.enable_backward_allreduce = False
        # We reset the compression errors when loading checkpoints for 3 reasons:
        # 1) The worker and server error at each GPU are distinct, so in current implementation
        # only rank 0's errors are saved in the checkpoint. Thus we have to reset the errors.
        # If we want to save them correctly we need O(num_gpu*model_size) memory in order to
        # gather all the error, which is a very large memory requirement. It's possible to save
        # them in a distributed way, but it will make the checkpoint saving/loading much more complicated.
        # 2) Even if we are able to save the compression errors correctly, you need to have the
        # exact same number of GPUs in order to load them correctly.
        # 3) We verified on BERT pre-training that occasionally resetting the compression error
        # at checkpoint loading does not affect the convergence.
        # However, please avoid frequent checkpoint loading which could break the error
        # compensation mechanism thus affect the convergence.
        for group in self.param_groups:
            for p in group['params']:
                if 'worker_error' in self.state[p]:
                    self.state[p].pop('worker_error')
                if 'server_error' in self.state[p]:
                    self.state[p].pop('server_error')

def pack_binary_tensor(tensor):
    # -1を0に、1を1に変換
    packed = (tensor + 1) // 2
    # 8ビットごとにパックするためのサイズの調整
    if packed.numel() % 8 != 0:
        # 必要なパディング量を計算
        padding_size = 8 - (packed.numel() % 8)
        # 末尾に0を追加してパディング
        packed = torch.cat([packed, torch.zeros(padding_size, dtype=packed.dtype, device=packed.device)])

    packed = packed.view(-1, 8)
    packed = (packed[:, 0] + packed[:, 1] * 2 + packed[:, 2] * 4 + packed[:, 3] * 8 + 
              packed[:, 4] * 16 + packed[:, 5] * 32 + packed[:, 6] * 64 + packed[:, 7] * 128)
    return packed.byte()

def unpack_binary_tensor(packed, original_shape):
    unpacked = torch.zeros(packed.numel() * 8, dtype=torch.uint8, device=packed.device)
    for i in range(8):
        unpacked[i::8] = (packed >> i) & 1
    unpacked = unpacked.to(torch.int8) * 2 - 1
    unpacked = unpacked[:original_shape.numel()].view(original_shape)
    return unpacked

def binary_quantize_allreduce(tensor):
    original_shape = tensor.shape
    
    quantized = torch.sign(tensor)
    # random_choice = torch.randint(0, 2, quantized.shape, dtype=quantized.dtype) * 2 - 1
    # random_choice = random_choice.to(quantized.device)
    # quantized = torch.where(quantized == 0, random_choice, quantized)
    packed = pack_binary_tensor(quantized)
    dist.all_reduce(packed)
    unpacked = unpack_binary_tensor(packed, original_shape)
    
    world_size = dist.get_world_size()
    result = unpacked.float() / world_size
    
    return result
