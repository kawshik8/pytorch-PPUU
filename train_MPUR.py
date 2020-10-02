import math
from collections import OrderedDict

import numpy
import os
import ipdb
import random
import torch
import torch.optim as optim
from os import path

import planning
import utils
from dataloader import DataLoader

from eval_policy import load_models

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train a policy / controller
#################################################
def start(what, nbatches, npred, split='train', return_per_instance_losses = False, threshold=0):
    train = True if what is 'train' else False
    model.train()
    model.policy_net.train()
    n_updates, grad_norm = 0, 0
    total_losses = dict(
        proximity=0,
        uncertainty=0,
        lane=0,
        offroad=0,
        action=0,
        policy=0,
    )
    if return_per_instance_losses:
        finetune_inputs = []
        total_losses = dict(
            proximity=[],
            uncertainty=[],
            lane=[],
            offroad=[],
            action=[],
            policy=[],
        )

    else:
        total_losses = dict(
            proximity=0,
            uncertainty=0,
            lane=0,
            offroad=0,
            action=0,
            policy=0,
        )

    for j in range(nbatches):
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(split, npred, cuda = (True if torch.cuda.is_available() and not opt.no_cuda else False))

        pred, actions = planning.train_policy_net_mpur(
            model, inputs, targets, car_sizes, n_models=10, lrt_z=opt.lrt_z,
            n_updates_z=opt.z_updates, infer_z=opt.infer_z, no_cuda=opt.no_cuda, return_per_instance_losses = return_per_instance_losses
        )
        pred['policy'] = pred['proximity'] + \
                         opt.u_reg * pred['uncertainty'] + \
                         opt.lambda_l * pred['lane'] + \
                         opt.lambda_a * pred['action'] + \
                         opt.lambda_o * pred['offroad']
        
        print(pred['policy'].shape)

        if not math.isnan(pred['policy'].item()):
            if train:
                optimizer.zero_grad()
                pred['policy'].backward()  # back-propagation through time!
                grad_norm += utils.grad_norm(model.policy_net).item()
                torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), opt.grad_clip)
                optimizer.step()

            if not return_per_instance_losses:
                for loss in total_losses: 
                    total_losses[loss] += pred[loss].item()
            else:
                for b_i in range(dataloader.opt.batch_size):
                    if pred['policy'][b_i] > threshold:
                        instance_loss = loss[b_i]
                        for loss in total_losses:
                            total_losses[loss].append(pred[loss].item())
                        finetune_inputs.append({"input":inputs[b_i], "action":actions[b_i], "target":targets[b_i], "id":ids[b_i], "car_size":car_sizes[b_i]})

            n_updates += 1
        else:
            print('warning, NaN')  # Oh no... Something got quite fucked up!
            ipdb.set_trace()

        if j == 0 and opt.save_movies and train:
            # save videos of normal and adversarial scenarios
            for b in range(opt.batch_size):
                state_img = pred['state_img'][b]
                state_vct = pred['state_vct'][b]
                utils.save_movie(opt.model_file + f'.mov/sampled/mov{b}', state_img, state_vct, None, actions[b])

        del inputs, actions, targets, pred

    if not return_per_instance_losses:
        for loss in total_losses: total_losses[loss] /= n_updates

    if train: print(f'[avg grad norm: {grad_norm / n_updates:.4f}]')

    if not return_per_instance_losses:
        return total_losses
    else:
        return finetune_inputs, total_losses

# if __name__=='__main__':

opt = utils.parse_command_line()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu')
# Define default device
if torch.cuda.is_available() and opt.no_cuda:
    print('WARNING: You have a CUDA device, so you should probably run without -no_cuda')

data_path = opt.data_dir + f'traffic-data/state-action-cost/data_{opt.dataset}_v0'
opt.model_file = path.join(opt.model_dir, 'policy_networks', 'MPUR-' + opt.policy)

if opt.method == 'train':
    # Create file_name
    
    utils.build_model_file_name(opt)

    os.system('mkdir -p ' + path.join(opt.model_dir, 'policy_networks'))

    # load the model

    model_path = path.join(opt.model_dir, opt.mfile)
    if path.exists(model_path):
        model = torch.load(model_path)
    elif path.exists(opt.mfile):
        model = torch.load(opt.mfile)
    else:
        raise runtime_error(f'couldn\'t find file {opt.mfile}')

    if not hasattr(model.encoder, 'n_channels'):
        model.encoder.n_channels = 3

    if type(model) is dict: model = model['model']
    model.opt.lambda_l = opt.lambda_l  # used by planning.py/compute_uncertainty_batch
    model.opt.lambda_o = opt.lambda_o  # used by planning.py/compute_uncertainty_batch
    if opt.value_model != '':
        value_function = torch.load(path.join(opt.model_dir, 'value_functions', opt.value_model)).to(opt.device)
        model.value_function = value_function

    # Create policy
    model.create_policy_net(opt)

    # Load normalisation stats
    stats = torch.load('traffic-data/state-action-cost/data_i80_v0/data_stats.pth')
    model.stats = stats  # used by planning.py/compute_uncertainty_batch
    if 'ten' in opt.mfile:
        p_z_file = opt.model_dir + opt.mfile + '.pz'
        p_z = torch.load(p_z_file)
        model.p_z = p_z

    model.policy_net.stats_d = {}
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            model.policy_net.stats_d[k] = v.to(opt.device)

else:
    (
        model,
        value_function,
        policy_network_il,
        policy_network_mper,
        data_stats
    ) = load_models(opt, data_path, opt.device)

    
optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)  # POLICY optimiser ONLY!

# Send to GPU if possible
model.to(opt.device)

if opt.learned_cost:
    print('[loading cost regressor]')
    model.cost = torch.load(path.join(opt.model_dir, opt.mfile + '.cost.model'))['model']

dataloader = DataLoader(None, opt, opt.dataset)
model.train()
model.opt.u_hinge = opt.u_hinge
planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred, no_cuda=opt.no_cuda)
model.eval()

print('[training]')
utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
n_iter = 0
losses = OrderedDict(
    p='proximity',
    l='lane',
    o='offroad',
    u='uncertainty',
    a='action',
    π='policy',
)

writer = utils.create_tensorboard_writer(opt)

if opt.method == 'train':
    for i in range(500):
        train_losses = start('train', opt.epoch_size, opt.npred, split='valid')
        with torch.no_grad():  # Torch, please please please, do not track computations :)
            valid_losses = start('valid', opt.epoch_size // 2, opt.npred, split='valid')

        if writer is not None:
            for key in train_losses:
                writer.add_scalar(f'Loss/train_{key}', train_losses[key], i)
            for key in valid_losses:
                writer.add_scalar(f'Loss/valid_{key}', valid_losses[key], i)

        n_iter += opt.epoch_size
        model.to('cpu')
        torch.save(dict(
            model=model,
            optimizer=optimizer.state_dict(),
            opt=opt,
            n_iter=n_iter,
        ), opt.model_file + '.model')
        if (n_iter / opt.epoch_size) % 10 == 0:
            torch.save(dict(
                model=model,
                optimizer=optimizer.state_dict(),
                opt=opt,
                n_iter=n_iter,
            ), opt.model_file + f'step{n_iter}.model')

        model.to(opt.device)

        log_string = f'step {n_iter} | '
        log_string += 'train: [' + ', '.join(f'{k}: {train_losses[v]:.4f}' for k, v in losses.items()) + '] | '
        log_string += 'valid: [' + ', '.join(f'{k}: {valid_losses[v]:.4f}' for k, v in losses.items()) + ']'
        print(log_string)
        utils.log(opt.model_file + '.log', log_string)

        if writer is not None:
            writer.close()

else:
    finetune_inputs, train_eval_losses = start('train', opt.epoch_size, opt.npred, split='train', return_per_instance_losses=True)
    print(len(finetune_inputs))
    print(train_eval_losses)


