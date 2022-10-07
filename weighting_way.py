from abc import abstractmethod

import numpy as np
import torch
from torch import nn



class GradCosine():
 
    def __init__(self, main_task, **kwargs):
        self.main_task = main_task
        self.cosine_similarity = nn.CosineSimilarity(dim=0)

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def get_grad_cos_sim(self, grad1, grad2):
        """Computes cosine simillarity of gradients after flattening of tensors.
        """

        flat_grad1 = self._flattening(grad1)
        flat_grad2 = self._flattening(grad2)

        cosine = nn.CosineSimilarity(dim=0)(flat_grad1, flat_grad2)

        return torch.clamp(cosine, -1, 1)

    def get_grad(self, losses, shared_parameters):
        """
        :param losses: Tensor of losses of shape (n_tasks, )
        :param shared_parameters: model that are not task-specific parameters
        :return:
        """

        main_loss = losses[self.main_task]
        aux_losses = torch.stack(tuple(l for i, l in enumerate(losses) if i != self.main_task))

        main_grad = torch.autograd.grad(main_loss, shared_parameters, retain_graph=True)
        # copy
        grad = tuple(g.clone() for g in main_grad)

        for loss in aux_losses:
            aux_grad = torch.autograd.grad(loss, shared_parameters, retain_graph=True)
            cosine = self.get_grad_cos_sim(main_grad, aux_grad)

            if cosine > 0:
                grad = tuple(g + ga for g, ga in zip(grad, aux_grad))

        return grad

    def backward(self, losses, shared_parameters, returns=True, **kwargs):
        shared_grad = self.get_grad(
            losses,
            shared_parameters=shared_parameters
        )
        loss = torch.sum(torch.stack(losses))
        loss.backward()
        # update grads for shared weights
        for p, g in zip(shared_parameters, shared_grad):
            p.grad = g

        if returns:
            return loss