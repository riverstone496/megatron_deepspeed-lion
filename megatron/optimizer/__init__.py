# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from deepspeed.accelerator import get_accelerator
if get_accelerator().device_name() == 'cuda':
    from apex.optimizers import FusedAdam as Adam
    from apex.optimizers import FusedSGD as SGD
else:
    from torch.optim import Adam
    from torch.optim import SGD


from megatron import get_args
from megatron.model import LayerNorm
from megatron import print_rank_0

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer

def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    args = get_args()

    weight_decay_params = {'params': [], 'name' : 'weight_decay_params'}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0, 'name': 'no_weight_decay_params'}
    
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                    if p is not None])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                    if p is not None and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                    if p is not None and n == 'bias'])
    return weight_decay_params, no_weight_decay_params

def get_megatron_optimizer(model, deepspeed=None):
    args = get_args()

    # Base optimizer.
    param_groups = _get_params_for_weight_decay_optimization(model)
    if args.create_moe_param_group:
        from deepspeed.moe.utils import is_moe_param, split_params_into_different_moe_groups_for_optimizer
        param_groups = split_params_into_different_moe_groups_for_optimizer(param_groups)
    print_rank_0(f'Optimizer = {args.optimizer} \n')
    if args.cpu_optimizer:
        assert args.optimizer == 'adam', 'CPU offloading is for Adam'
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay)
    else:
        if args.optimizer == 'adam':
            optimizer = Adam(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_eps)
        elif args.optimizer == 'sgd':
            optimizer = SGD(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            momentum=args.sgd_momentum)
        elif args.optimizer == 'zerooneadam':
            from .onebit.zoadam import ZeroOneAdam
            optimizer = ZeroOneAdam(param_groups,
                            deepspeed=deepspeed,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_eps)
        elif args.optimizer == 'onebitadam':
            from .onebit.adam import OnebitAdam
            optimizer = OnebitAdam(param_groups,
                            deepspeed=deepspeed,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_eps,
                            freeze_step=args.momentum_freeze_step)
        elif args.optimizer == 'lion':
            from .standard.lion import Lion
            optimizer = Lion(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2))
        elif args.optimizer == 'lion_all':
            from .standard.lion_all import LionAll
            optimizer = LionAll(param_groups,
                            deepspeed=deepspeed,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2))
        elif args.optimizer == 'lion_all2':
            from .onebit.lion_all import LionAll
            optimizer = LionAll(param_groups,
                            deepspeed=deepspeed,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2))
        elif args.optimizer == 'mvlion':
            from .onebit.lion import MVLion
            optimizer = MVLion(param_groups,
                            deepspeed=deepspeed,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            freeze_step=args.momentum_freeze_step)
        elif args.optimizer == 'mvlion2':
            from .onebit.lion_optimized import MVLion
            optimizer = MVLion(param_groups,
                            deepspeed=deepspeed,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            freeze_step=args.momentum_freeze_step)
        elif args.optimizer == 'onebitlion':
            from .onebit.onebitlion import OnebitLion
            optimizer = OnebitLion(param_groups,
                            deepspeed=deepspeed,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            freeze_step=args.momentum_freeze_step)
        else:
            raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    if args.deepspeed:
        return optimizer

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    if args.fp16 or args.bf16:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        return Float16OptimizerWithFloat16Params(optimizer,
                                                 args.clip_grad,
                                                 args.log_num_zeros_in_grad,
                                                 params_have_main_grad,
                                                 args.bf16,
                                                 grad_scaler)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad)
