import torch
import torch.nn as nn
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector


def L2SP_Fisher(model, w0_dic, new_layers, num_lowlrs, inp):
    existing_l2_reg = None
    new_l2_reg = None

    for name, w in model.named_parameters():
        # print(name)
        # if 'weight' not in name:  # I don't know if that is true: I was told that Facebook regularized biases too.
        #    continue
        # if 'downsample.1' in name:  # another bias
        #    continue
        if 'bn' in name:  # bn parameters
            continue
        # fc layers
        if 'linear' in name:
            if new_l2_reg is None:
                new_l2_reg = torch.pow(w, 2).sum() / 2
            else:
                new_l2_reg += torch.pow(w, 2).sum() / 2
        # new layers must be l2 not L2SP as there is no w0
        elif (int(name.split('.')[1]) >= (22 - new_layers)):
            if new_l2_reg is None:
                new_l2_reg = torch.pow(w, 2).sum() / 2
            else:
                new_l2_reg += torch.pow(w, 2).sum() / 2
            # print('new: ' + str(new_l2_reg) + ' ' + str(name))
        else:
            # print(name)
            w0 = w0_dic[name].data
            w0 = w0.cuda()
            # split layers the same way as high and low learning rates so we can do different things with them
            # note - the number of weights in the fc layer is orders of magnitude great than those in the conv layer
            # also lower layers should be more transferable so keeping them closer to their pretrained values is more likely to be optimal
            if (int(name.split('.')[1]) < (num_lowlrs)):
                if existing_l2_reg is None:
                    existing_l2_reg = 10 * (torch.pow(w - w0, 2).sum() / 2)
                else:
                    existing_l2_reg += 5 * (torch.pow(w - w0, 2).sum() / 2)
                # print('existing: ' + str(existing_l2_reg) + ' ' + str(name))
            else:
                if new_l2_reg is None:
                    new_l2_reg = torch.pow(w, 2).sum() / 2
                else:
                    new_l2_reg += torch.pow(w, 2).sum() / 2
                # if existing_l2_reg is None:
                #    existing_l2_reg = (torch.pow(w-w0, 2).sum()/2)
                # else:
                #    existing_l2_reg += (torch.pow(w-w0, 2).sum()/2)

    # print('existing: ' + str(existing_l2_reg))
    # print('new: ' + str(new_l2_reg))
    l2_reg = existing_l2_reg * 0.004 + new_l2_reg * 0.0005

    # Compute Fisher loss
    fisher_reg = EWC(model, inp).penalty(model)
    l2_reg += fisher_reg * 0.004
    # print(l2_reg)

    return l2_reg

class EWC(object):
    def __init__(self, model: nn.Module, train_set):
        self.model = model
        self.train_set = train_set
        self.Fisher, self.v0 = self.compute_fisher(self.model, PMatKFA)

    def compute_fisher(self, model, Representation):
        fisher_set = deepcopy(self.train_set)
        fisher_loader = DataLoader(fisher_set, batch_size=50, shuffle=False, num_workers=6)
        F_diag = FIM(model=model,
                     loader=fisher_loader,
                     representation=Representation,
                     n_output=30,
                     variant='classif_logits',
                     device='cuda')

        v0 = PVector.from_model(model).clone().detach()

        return F_diag, v0

    def penalty(self, model: nn.Module):
        v = PVector.from_model(model)
        regularization_loss = self.Fisher.vTMv(v - self.v0)

        return regularization_loss
