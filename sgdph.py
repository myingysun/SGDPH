# *
import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class sgdph(Optimizer):
    """ Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 5e-4)

        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
    """

    def __init__(self, params, lr=0.1,
                 weight_decay=0, momentum=0.9, eps=0.0001,tau=0.001, single_gpu=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)

        self.single_gpu = single_gpu
        self.tau =tau
        self.steps=0
        self.eps=eps
        super(sgdph, self).__init__(params, defaults)
        self.init_M()        
        
    def init_M(self):
        M=0
        for group in self.param_groups:
            for p in group['params']:
               if p.dim()==1:
                  M+=1
        self.M=M      
        

    def get_trace(self, params, grads, k):
        """
        compute the diagonal hessian with respect to a specific BN layer
        :return: a list of tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                                   '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                                   '\t\t\t  set to True.')

        # Define the vector to get the diagonal elements.
        j = 0
        e = []
        grad_update = []
        params_update = []
        for (p, grad) in zip(params, grads):          
            if p.dim()==1:
               j+=1
               if j==k:
                  e.append(torch.ones_like(p))
                  grad_update.append(grad)
                  params_update.append(p)    
               elif j>k:               
                  e.append(torch.zeros_like(p))
                  grad_update.append(grad)
                  params_update.append(p)                
            else:
               if j>=k: 
                  e.append(torch.zeros_like(p))
                  grad_update.append(grad)
                  params_update.append(p)                  

        # compute the diagonal elements by back propagation
        if not e:
            pass
        else:
            bn_diag_k_update = torch.autograd.grad(
                grad_update,
                params_update,
                grad_outputs=e,
                only_inputs=True,
                retain_graph=True)
        return bn_diag_k_update[0].abs()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []
        Iterlayer=self.steps%self.M+1
        self.steps+=1        
        # Flatten groups into lists, so that get_trace can be called with lists of parameters and grads
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # for k-th iteration get the Hessian diagonal
        bn_diag_k = self.get_trace(params, grads, Iterlayer)

        iterlayer=0
        for (p,group, grad) in zip(params,groups, grads):
                weight_decay = group['weight_decay']
                param_state = self.state[p]
                momentum = group['momentum'] 
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1)
                    #hgrad = buf                               
                if p.dim()==1:
                   if 'diag' not in param_state:
                     param_state['diag']=torch.zeros_like(p, memory_format=torch.preserve_format)                
                   iterlayer+=1                   
                   if iterlayer==Iterlayer:
                         param_state['diag'].mul_(group['momentum']).add_(bn_diag_k, alpha=1-group['momentum'])       
                   hgrad = (1/(param_state['diag']+self.eps)) * buf*(1-group['momentum']**(int(self.steps/self.M)+1))*self.tau  
                   if iterlayer==self.M:
                      hgrad = buf                           
                else:
                   hgrad = buf                
                # Apply weight decay and momentum for descent direction (the same as SGD)
                if weight_decay != 0:
                    hgrad.add_(p, alpha=weight_decay)
                # make update
                p.data = p.data - group['lr'] * hgrad
        return loss