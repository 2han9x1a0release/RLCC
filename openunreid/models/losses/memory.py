
import torch
import torch.nn.functional as F
from torch import autograd, nn

from ...utils.dist_utils import all_gather_tensor

try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd
    class HM(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None
except:
    class HM(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )

class HybridMemory(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
        self.register_buffer("feature_weights", torch.ones(num_memory).float())



    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))

    @torch.no_grad()
    def _update_feature_weights(self, feature_weights):
        self.feature_weights.data.copy_(feature_weights.float().to(self.feature_weights.device))


    def forward(self, results, indexes, cur_epoch, sample_softlabels = 0, batch_indexs_labels = 0, alpha_weight = 1.0):
        inputs = results["feat"]
        inputs = F.normalize(inputs, p=2, dim=1)

        batch_sample_weights = self.feature_weights[indexes]
        batch_sample_weights = batch_sample_weights.reshape((batch_sample_weights.size(0),1))

        if cur_epoch != 0:
            batch_soft_labels = sample_softlabels[batch_indexs_labels].cuda()

        inputs = hm(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        batch_sample_weights = self.feature_weights[indexes]

        sim = torch.zeros(labels.max() + 1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        # masked_sim -> input probs
        log_probs = torch.log(masked_sim + 1e-6)

        Loss1 = F.nll_loss(log_probs, targets) 

        # Label Smoothing regularazation
        N = log_probs.size(0)  # batch_size
        C = log_probs.size(1)  # number of classes
        alpha = 0.0
        smoothed_labels = torch.full(size=(N, C), fill_value = alpha / (C - 1)).cuda()
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1-alpha)
        Label_Smoothed_Loss = (-smoothed_labels.detach() * log_probs).mean(0).sum()
        beta = 0.0
        
        # Bootstraping Loss
        Loss4 = (-(masked_sim + 1e-6).detach() * log_probs).mean(0).sum()
        if cur_epoch == 0:
            return 0.0 * Label_Smoothed_Loss + 1.0 * Loss1 + 0.0 * Loss4
        else:
            Loss2 = (-batch_soft_labels.detach() * log_probs).mean(0).sum()
            refine_soft_labels = (1.0 - beta) * batch_soft_labels + beta * smoothed_labels
            Loss3 = (-refine_soft_labels.detach() * log_probs).mean(0).sum()
            return 0.0 * Label_Smoothed_Loss + alpha_weight * Loss1 + (1.0 - alpha_weight) * Loss2 + 0.0 * Loss3 + 0.0 * Loss4


