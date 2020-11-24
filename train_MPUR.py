import math
from collections import OrderedDict

import numpy as np

import os
import ipdb
import random
import torch
import torch.optim as optim
from os import path
from tqdm import tqdm

import planning
import utils
from dataloader import DataLoader
import matplotlib.pyplot as plt

from eval_policy import load_models
import time
import pickle 
from datetime import datetime
# from tensorboardX import SummaryWriter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

total_steps = 0

#################################################
# Train a policy / controller
#################################################
def start(what, nbatches, npred, split='train', return_per_instance_values=False, threshold=0):
    train = True if what is 'train' else False
    evaluate = True if what is 'eval' else False
    finetune_train = True if split is 'finetune_train' else False
    finetune_sim = True if split is 'finetune_sim' else False
    
    model.train()
    model.policy_net.train()
    n_updates, grad_norm = 0, 0
    
    if return_per_instance_values:
        total_losses = dict(
            proximity=[],
            uncertainty=[],
            lane=[],
            offroad=[],
            action=[],
            policy=[],
            episode_timestep_pairs=[],
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
        
    if evaluate:
        episode_cost_progression = {}

    iterable = range(nbatches)
    if evaluate or finetune_train:
        total_instances = dataloader.get_total_instances(split,what)
        print(f"total_instances in {split}: {total_instances}")
        iterable = range(0,total_instances,opt.batch_size)
#         nbatches = None
        
    step = 0
    for j in iterable:
#             print("j:",j,n_updates)
#         with tqdm(total=len(iterable)) as progress_bar:
#             start = time.time()
            if not evaluate:
                inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(split, npred, cuda = (True if torch.cuda.is_available() and not opt.no_cuda else False), all_batches=(True if finetune_train else False))
            else:
                e_index, inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(split, npred, return_episode_index=True, cuda = (True if torch.cuda.is_available() and not opt.no_cuda else False), all_batches=True if finetune_train else False, randomize=(True if (finetune_train or finetune_sim) else False) )
#                 print(np.unique(e_index))#, type(e_index))
                if -1 in e_index[:,0]:
                    print("breaking now")
                    break

            pred, actions = planning.train_policy_net_mpur(
                model, inputs, targets, car_sizes, n_models=10, lrt_z=opt.lrt_z,
                n_updates_z=opt.z_updates, infer_z=opt.infer_z, no_cuda=opt.no_cuda, return_per_instance_values = (True if evaluate else False)
            )
            pred['policy'] = pred['proximity'] + \
                             opt.u_reg * pred['uncertainty'] + \
                             opt.lambda_l * pred['lane'] + \
                             opt.lambda_a * pred['action'] + \
                             opt.lambda_o * pred['offroad']

#             print(torch.mean(pred['policy']))
#             print(pred['policy'].shape)
#             print("time for loading batches and forward pass: ", time.time() - start)
#             start = time.time()
            
            if (not evaluate and not math.isnan(pred['policy'].item())) or (evaluate and not math.isnan(torch.mean(pred['policy']).item())):
                if train:
                    optimizer.zero_grad()
                    pred['policy'].backward()  # back-propagation through time!
                    grad_norm += utils.grad_norm(model.policy_net).item()
                    torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), opt.grad_clip)
                    optimizer.step()
                n_updates += 1

                if not evaluate:
                    for loss in total_losses: 
                        
                        total_losses[loss] += pred[loss]
                        global total_steps
                        writer.add_scalar(f'BatchLoss/{what}_{loss}', pred[loss].detach().cpu().item(), total_steps)
                        total_steps += 1
#                     print(total_losses)
#                         if what == 'valid':
#                             print(loss,total_losses[loss])
                else:
                    for b_i in range(len(actions)):
#                         print(pred['policy'].shape,len(ids),len(ids[0]),type(e_index))
                        instance_loss = torch.mean(pred['policy'][b_i]).detach().cpu().item()

                        if e_index[b_i][0] not in episode_cost_progression:
                            episode_cost_progression[e_index[b_i][0]] = {key:[] for key in total_losses if "episode" not in key}
    
                        if instance_loss >= threshold:
                            episode_index, timestep = e_index[b_i]
                                
                            for loss in total_losses:
                                if loss!= 'episode_timestep_pairs':
                                    total_losses[loss].append(torch.mean(pred[loss][b_i]).detach().cpu())
                                    episode_cost_progression[e_index[b_i][0]][loss].append(torch.mean(pred[loss][b_i]).detach().cpu())
                                else:
                                    total_losses[loss].append([episode_index,timestep,instance_loss])
                                    
                                
                            
                            
#                             print(type(dataloader.finetune_dict))
#                             print(dataloader.finetune_dict)
#                             if episode_index not in dataloader.finetune_dict: 
#                                 dataloader.finetune_dict[episode_index] = [] 
                            
#                             nframes = opt.npred + opt.ncond
#                             min_range = max(0,timestep-nframes)
#                             max_range = min(len(dataloader.images[episode_index]),timestep+100)-50
#                             for frame_index in range(min_range,max_range,opt.finetune_nframes_overlap):
#                                 if frame_index not in dataloader.finetune_dict[episode_index]:
#                                     dataloader.finetune_dict[episode_index].append(frame_index)
                            
#                             else:
# #                                 print(type(inputs),type(inputs[0]),type(inputs[0][b_i]))
                                
#                                 if finetune_inputs["inputs"]:
#                                     for f_i, input_i in enumerate(finetune_inputs["inputs"]):
#                                         finetune_inputs["inputs"][input_i].append(inputs[f_i][b_i:b_i+1])
#                                     for f_i, input_i in enumerate(finetune_inputs["targets"]):
#                                         finetune_inputs["targets"][input_i].append(targets[f_i][b_i:b_i+1])
#                                 else:
#                                     finetune_inputs["inputs"] = {f_i : [inputs[f_i][b_i:b_i+1]] for f_i in range(len(inputs))}
#                                     finetune_inputs["targets"] = {f_i : [targets[f_i][b_i:b_i+1]] for f_i in range(len(targets))}
                                    
# #                                 print(type(car_sizes), len(car_sizes), type(car_sizes[0]), len(car_sizes[0]))
                                
#                                 finetune_inputs["car_sizes"].append(car_sizes[b_i:b_i+1])
                                
#                                 if len(finetune_inputs["car_sizes"]) == opt.batch_size:
# #                                     print([torch.cat(finetune_inputs["inputs"][input_i]).shape for input_i in finetune_inputs["inputs"]])
# #                                     for row in inputs:
# #                                         print(row.shape)
# #                                     print([torch.cat(finetune_inputs["targets"][input_i]).shape for input_i in finetune_inputs["targets"]])
# #                                     for row in targets:
# #                                         print(row.shape)
#                                     finetune_inputs["inputs"] = [torch.cat(finetune_inputs["inputs"][input_i]) for input_i in finetune_inputs["inputs"]]
#                                     finetune_inputs["targets"] = [torch.cat(finetune_inputs["targets"][input_i]) for input_i in finetune_inputs["targets"]]
#                                     finetune_inputs["car_sizes"] = torch.cat(finetune_inputs["car_sizes"])
# #                                     t1,t2,t3 = finetune_inputs["targets"]
# #                                     print(type(targets), len(targets), type(targets[0]), len(targets[0]))
# #                                     print(type(finetune_inputs["targets"]),len(finetune_inputs["targets"]),type(finetune_inputs["targets"][0]), len(finetune_inputs["targets"][0]))
#                                     pred, actions = planning.train_policy_net_mpur(
#                                         model, finetune_inputs["inputs"], finetune_inputs["targets"], finetune_inputs["car_sizes"], n_models=10, lrt_z=opt.lrt_z, n_updates_z=opt.z_updates, infer_z=opt.infer_z, no_cuda=opt.no_cuda )
                                       
#                                     pred['policy'] = pred['proximity'] + \
#                                                      opt.u_reg * pred['uncertainty'] + \
#                                                      opt.lambda_l * pred['lane'] + \
#                                                      opt.lambda_a * pred['action'] + \
#                                                      opt.lambda_o * pred['offroad']
            
#                                     for loss in finetune_losses: 
#                                         finetune_losses[loss] += pred[loss]
                    
#                                     optimizer.zero_grad()
#                                     pred['policy'].backward()  # back-propagation through time!
#                                     grad_norm += utils.grad_norm(model.policy_net).item()
#                                     torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), opt.grad_clip)
#                                     optimizer.step()
                                    
#                                     n_updates += 1
#                                     print(f"update no: {n_updates}")
                                    
                                    
#                                     finetune_inputs["inputs"] = {}
#                                     finetune_inputs["car_sizes"] = []
#                                     finetune_inputs["targets"] = {}
                                   
                                    

                
            else:
                print('warning, NaN')  # Oh no... Something got quite fucked up!
                ipdb.set_trace()

            if j == 0 and opt.save_movies and train:
                # save videos of normal and adversarial scenarios
                for b in range(opt.batch_size):
                    state_img = pred['state_img'][b]
                    state_vct = pred['state_vct'][b]
                    utils.save_movie(opt.model_file + f'.mov/sampled/mov{b}', state_img, state_vct, None, actions[b])

#             step += len(actions)
#             progress_bar.update(step)
            del inputs, actions, targets, pred
    
            if n_updates==nbatches and not evaluate:
#                 del dataloader[split]
                
                break
#             print("time for saving loss values for calc stats later: ", time.time() - start)

    if not evaluate:
        for loss in total_losses: total_losses[loss] /= n_updates
#     print(total_losses)

    if train or finetune_train or finetune_sim: print(f'[avg grad norm: {grad_norm / n_updates:.4f}]')

        
    if evaluate: pickle.dump(episode_cost_progression, open("policy_loss_stats/episode_cost_progression.pkl",'wb+'))
#     if finetune:
#         return finetune_losses 
#     print("final j value: ", j)
    return total_losses


# if __name__=='__main__':

def evaluate_policy(split='train', stats_filename='stats_policy_model.txt', all_batches=False):
    
    print(f"Evaluating model on batches of {split} and saving the stats in {stats_filename.split('.')[0]}_{split}.txt")
    with torch.no_grad():
        train_eval_losses = start(what='eval', npred=opt.npred, split=split, return_per_instance_values=True, nbatches = opt.epoch_size)
#     print("no of high error cases: ", len(train_eval_losses['episode']))
#     print("mean/max/min of each error type")
#     print(dataloader.n_episodes)
    stats_filename = stats_filename.split(".")[0]
    
    episode_wise_loss_stats = {}
    file = open(f"{stats_filename}_{split}_o{opt.eval_nframes_overlap}.txt",'w+')
    for loss in train_eval_losses:
        episode_stats = {}
        
        if loss != 'episode_timestep_pairs':
#             print(loss)
            file.write("\n\n" + str(loss) + "\n")
            for row in range(len(train_eval_losses[loss])):
                episode_index = train_eval_losses['episode_timestep_pairs'][row][0].astype(int)
                if episode_index in episode_stats:
#                     if loss == 'policy':
#                         print("sample losses:",train_eval_losses[loss][row])
                    episode_stats[episode_index]["count"] += 1
                    episode_stats[episode_index]["max"] = max(episode_stats[episode_index]["max"], train_eval_losses[loss][row].item())
                    episode_stats[episode_index]["min"] = min(episode_stats[episode_index]["min"], train_eval_losses[loss][row].item())
                    episode_stats[episode_index]["loss"] += [train_eval_losses[loss][row].item()]
                else:
                    episode_stats[episode_index] = {"count":1,"mean":0,"max":max(0,train_eval_losses[loss][row].item()),"min":min(10000,train_eval_losses[loss][row].item())}
                    episode_stats[episode_index]["loss"] = [train_eval_losses[loss][row].item()]

            
            total_loss = []
#             print("episode wise stats")
            file.write("\nepisode wise stats:\n")
            for episode_i in episode_stats:
                
                episode_stats[episode_i]["mean"] = np.mean(episode_stats[episode_i]["loss"])
                episode_stats[episode_i]["median"] = np.median(episode_stats[episode_i]["loss"])
#                 print(episode_i,episode_stats[episode_i]["count"],episode_stats[episode_i]["mean"].item(), episode_stats[episode_i]["min"].item(), episode_stats[episode_i]["max"].item())
                file.write(str(episode_i) + ", " + str(episode_stats[episode_i]["count"]) + ", " + str(episode_stats[episode_i]["median"])+ ", " + str(episode_stats[episode_i]["mean"]) + ", " + str(episode_stats[episode_i]["min"]) + ", " + str(episode_stats[episode_i]["max"]) + "\n")
    #             episode_wise_loss[loss][episode_i] /= (min(dataloader.images[episode_i].size(0), dataloader.states[episode_i].size(0))-(opt.npred + opt.ncond))
    #             print(min(dataloader.images[episode_i].size(0), dataloader.states[episode_i].size(0))-(opt.npred + opt.ncond))
    #             print(len(episode_wise_loss),len(episode_wise_loss[loss][episode_i]))
    #             print(episode_i, np.mean(episode_wise_loss[loss][episode_i]), np.min(episode_wise_loss[loss][episode_i]), np.max(episode_wise_loss[loss][episode_i]))
#                 print(len(episode_stats[episode_i]["loss"]))
                total_loss += episode_stats[episode_i]["loss"]
#             print(total_loss,len(total_loss))
#             plt.hist(np.log(total_loss), linewidth=1, color='b')
#             plt.savefig(f"policy_loss_stats/{loss}_loss_means.jpg")
            
#             print("total stats")
            file.write("\ntotal stats:")
#             print(np.mean(total_loss), np.min(total_loss), np.max(total_loss))
            file.write(str(np.mean(total_loss)) + ", " + str(np.min(total_loss)) + ", " + str(np.max(total_loss)) + ", " + str(np.median(total_loss)) + ", " + str(np.sort(total_loss)[int(opt.negative_mining_topn*len(total_loss))]) + "\n")
    
#             print("length of total loss check: ",len(total_loss))
            if loss == 'policy':
                policy_loss_stats = [np.mean(total_loss), np.std(total_loss), np.min(total_loss), np.max(total_loss), np.median(total_loss), np.sort(total_loss)[int(opt.negative_mining_topn*len(total_loss))]]     
                
#             print(np.sort(total_loss))
#             print(int(opt.negative_mining_topn*len(total_loss)))
            episode_wise_loss_stats[loss] = {"episode_wise_stats" : episode_stats, "total_stats" : [np.mean(total_loss), np.std(total_loss), np.min(total_loss), np.max(total_loss), np.median(total_loss), np.sort(total_loss)[int(opt.negative_mining_topn*len(total_loss))-1]], "values" : total_loss, "topn":opt.negative_mining_topn}
        
#     episode_wise_stats = open(opt.episode_wise_policy_stats, 'w+')
    pickle.dump(episode_wise_loss_stats, open(f"{opt.episode_wise_policy_stats.split('.')[0]}_{split}_o{opt.eval_nframes_overlap}.pkl", 'wb+')) 
#     episode_wise_stats.write(str_stats)
#     episode_wise_stats.close()
    file.close()
    
    dataloader.finetune_dict = {}
    episode_timestep_pairs = train_eval_losses['episode_timestep_pairs']
    for row in episode_timestep_pairs:
            
            if row[-1] >= policy_loss_stats[-1]:
                 
                episode_index, timestep = row[0:2]
                if episode_index not in dataloader.finetune_dict: 
                    dataloader.finetune_dict[episode_index] = [] 
                            
                nframes = opt.ncond + opt.npred
                print(nframes)
#                 print(row[-1], policy_loss_stats[0], timestep)
                min_range = max(0,timestep-nframes)
                max_range = min(len(dataloader.images[episode_index]),timestep+(2*nframes))-nframes
                for frame_index in range(min_range,max_range,opt.finetune_nframes_overlap):
                    if frame_index not in dataloader.finetune_dict[episode_index]:
                        dataloader.finetune_dict[episode_index].append(frame_index)
                        
#     finetune_dict_file = 
    finetune_dict_and_triplets = {"finetune_overlap": opt.finetune_nframes_overlap, "finetune_dict": dataloader.finetune_dict, "episode_timestep_ploss_triplets": train_eval_losses['episode_timestep_pairs'], "top_k": opt.negative_mining_topn}
    pickle.dump(finetune_dict_and_triplets, open(opt.finetune_dict_file, 'wb+'))
#     finetune_dict_file.write(str_dict)
#     finetune_dict_file.close()
    
#     print(dataloader.finetune_dict)
    total = 0
    for episode in dataloader.finetune_dict:
        total += len(dataloader.finetune_dict[episode])
    print("Negatives num_episodes: ",len(list(dataloader.finetune_dict.keys())),"\tnum_instances:",total)
    return policy_loss_stats
    
    
date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

opt = utils.parse_command_line()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu')
# Define default device
if torch.cuda.is_available() and opt.no_cuda:
    print('WARNING: You have a CUDA device, so you should probably run without -no_cuda')

data_path = os.path.join(opt.data_dir,f'state-action-cost/data_{opt.dataset}_v0')
# print("data_dir:", opt.data_dir)
# print("data_path:",data_path)
opt.model_file = path.join(opt.model_dir, 'policy_networks', 'MPUR-' + opt.policy)

if opt.training_method == 'train':
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
#     print(opt.dataset)
#     print("data_path",data_path)
    stats = torch.load(data_path + '/data_stats.pth')
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

if not hasattr("model.encoder","n_channels"):
    model.encoder.n_channels = 3
    
model.opt.lambda_l = opt.lambda_l  # used by planning.py/compute_uncertainty_batch
model.opt.lambda_o = opt.lambda_o  # used by planning.py/compute_uncertainty_batch
    
optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)  # POLICY optimiser ONLY!
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True) 

# Send to GPU if possible
model.to(opt.device)

if opt.learned_cost:
    print('[loading cost regressor]')
    model.cost = torch.load(path.join(opt.model_dir, opt.mfile + '.cost.model'))['model']

all_batches = (True if opt.training_method == 'finetune_train' else False)
dataloader = DataLoader(None, opt, data_path, all_batches=all_batches)
model.train()
model.opt.u_hinge = opt.u_hinge
planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred, no_cuda=opt.no_cuda)
model.eval()

print('[training]')
log_string = f'[training]'
utils.log(opt.model_file + f'_{date_str}.log', log_string)
utils.log(opt.model_file + f'_{date_str}.log', f'[job name: {opt.model_file}]')
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

# print("\n return per instance values: ", opt.training_method)

if opt.training_method == 'train':
    for i in range(500):
        
        train_losses = start('train', opt.epoch_size, opt.npred, split='train')
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
        print("total_steps: ",total_steps)
        utils.log(opt.model_file + f'_{date_str}.log', log_string)

        if writer is not None:
            writer.close()

elif opt.training_method == 'finetune_train':
#     tbx = SummaryWriter("./policy_loss_stats/tensorboard_logs")
#     total_instances = dataloader.get_total_instances('train','finetune') 
#     print("total instances: ", total_instances)
#     step = 0
#     for i in range(0,total_instances,opt.batch_size):
# #     while len(list(dataloader.episode_dict['train'].keys())) > 0:
#         with tqdm(total=total_instances) as progress:
#             outputs = dataloader.get_batch_fm('train', opt.npred, return_episode_index=True, cuda = (True if torch.cuda.is_available() and not opt.no_cuda else False), all_batches=True, randomize=True)
        
# #             print(outputs[0])
# #             if None in outputs[0]:
# #                 break
# #             print(outputs[0][0])
# #             print("no of left out keys: ", len(list(dataloader.episode_dict['train'].keys())), " out of ", len(dataloader.train_indx))
#             print(i,outputs[2].shape)
#             step += len(outputs[2])
#             progress.update(step)
        
    
    
    if os.path.exists(f"{opt.episode_wise_policy_stats.split('.')[0]}_train_o{opt.eval_nframes_overlap}.pkl") and os.path.exists(f"{opt.finetune_dict_file.split('.')[0]}.pkl"):
        stats = pickle.load(open(f"{opt.episode_wise_policy_stats.split('.')[0]}_train_o{opt.eval_nframes_overlap}.pkl",'rb'))
#         print(stats.keys(), stats['policy'].keys())
        policy_stats = stats['policy']['total_stats']
#         print(policy_stats)
#         print(policy_stats[-1])
#         print(int(opt.negative_mining_topn*len(stats['policy']['values'])),len(stats['policy']['values']))
        
        if opt.negative_mining_topn != stats['policy']["topn"]:
            policy_stats[-1] = np.sort(stats['policy']['values'])[int(opt.negative_mining_topn*len(stats['policy']['values']))]
    
        fdict = pickle.load(open(opt.finetune_dict_file,'rb'))
        if fdict["finetune_overlap"]==opt.finetune_nframes_overlap and "top_k" in fdict and fdict["top_k"]==opt.negative_mining_topn:
            finetune_dict = fdict["finetune_dict"]
    #         print(finetune_dict)
            dataloader.finetune_dict = finetune_dict
            
        else:
            dataloader.finetune_dict = {}
            episode_timestep_pairs = fdict["episode_timestep_ploss_triplets"]
            for row in episode_timestep_pairs:
            
                if row[-1] >= policy_stats[-1]:

                    episode_index, timestep = row[0:2]
                    if episode_index not in dataloader.finetune_dict: 
                        dataloader.finetune_dict[episode_index] = [] 

                    nframes = opt.negative_sample_neighbourhood
    #                 print(row[-1], policy_loss_stats[0], timestep)
                    min_range = max(0,timestep-nframes)
                    max_range = min(len(dataloader.images[episode_index]),timestep+(2*nframes))-nframes
                    for frame_index in range(min_range,max_range,opt.finetune_nframes_overlap):
                        if frame_index not in dataloader.finetune_dict[episode_index]:
                            dataloader.finetune_dict[episode_index].append(frame_index)
            
#         
        
        
        total = 0
        for episode in dataloader.finetune_dict:
            total += len(dataloader.finetune_dict[episode])
        print("Negatives num_episodes: ",len(list(dataloader.finetune_dict.keys())),"\tnum_instances:",total)
        log_string = f"Negatives num_episodes: {len(list(dataloader.finetune_dict.keys()))} \tnum_instances: {total}"
        utils.log(opt.model_file + f'_{date_str}.log', log_string)
    else:
        policy_stats = evaluate_policy(split='train')
        
        
#         stats = pickle.load(open(f"{opt.episode_wise_policy_stats.split('.')[0]}_{split}_o{opt.eval_nframes_overlap}.pkl",'rb'))
#         print(stats)
#         finetune_dict = pickle.load(open(opt.finetune_dict_file,'rb'))
    
#         print(finetune_dict)
#     print(dataloader.finetune_dict)
    print("Policy_stats:",policy_stats)
    
    print("Starting Finetuning Process")
    best_loss = 1000000
    patience = 0
    for i in range(500):
        train_losses = start('train', opt.epoch_size, opt.npred, split='finetune_train', threshold = 0)
        with torch.no_grad():  # Torch, please please please, do not track computations :)
            valid_losses = start('valid', opt.epoch_size // 2, opt.npred, split='valid')
            scheduler.step(valid_losses['policy'])

        if writer is not None:
            for key in train_losses:
                writer.add_scalar(f'Loss/finetune_{key}', train_losses[key], i)
            for key in valid_losses:
                writer.add_scalar(f'Loss/valid_{key}', valid_losses[key], i)

        n_iter += opt.epoch_size
        model.to('cpu')
        torch.save(dict(
            model=model,
            optimizer=optimizer.state_dict(),
            opt=opt,
            n_iter=n_iter,
            finetune_dict=dataloader.finetune_dict,
        ), opt.model_file + f'topn={opt.negative_mining_topn}_stride={opt.eval_nframes_overlap}_lr={opt.lrt}_rlr_plateau_finetuned.model')
        if (n_iter / opt.epoch_size) % 10 == 0:
            torch.save(dict(
                model=model,
                optimizer=optimizer.state_dict(),
                opt=opt,
                n_iter=n_iter,
                finetune_dict=dataloader.finetune_dict,
            ), opt.model_file + f'topn={opt.negative_mining_topn}_stride={opt.eval_nframes_overlap}_lr={opt.lrt}_rlr_plateau_finetuned_step{n_iter}.model')

        model.to(opt.device)

        log_string = f'step {n_iter} | '
        log_string += 'train: [' + ', '.join(f'{k}: {train_losses[v]:.4f}' for k, v in losses.items()) + '] | '
        log_string += 'valid: [' + ', '.join(f'{k}: {valid_losses[v]:.4f}' for k, v in losses.items()) + ']'
#         print(log_string)
        utils.log(opt.model_file + f'_{date_str}.log', log_string)

        if valid_losses['policy'] < best_loss:
            best_loss = valid_losses['policy']
            patience = 0
        else:
            patience += 1
            
        if patience >= opt.finetune_earlystop_patience:
            log_string = f'early stopping at {i}\'th epoch'
            utils.log(opt.model_file + f'_{date_str}.log', log_string)
            break
            
        if writer is not None:
            writer.close()
            
elif opt.training_method == 'finetune_sim'
    for i in range(500):
        train_losses = start('train', opt.epoch_size, opt.npred, split='finetune_sim', threshold = 0)
        with torch.no_grad():  # Torch, please please please, do not track computations :)
            valid_losses = start('valid', opt.epoch_size // 2, opt.npred, split='valid')
            scheduler.step(valid_losses['policy'])

        if writer is not None:
            for key in train_losses:
                writer.add_scalar(f'Loss/finetune_{key}', train_losses[key], i)
            for key in valid_losses:
                writer.add_scalar(f'Loss/valid_{key}', valid_losses[key], i)

        n_iter += opt.epoch_size
        model.to('cpu')
        torch.save(dict(
            model=model,
            optimizer=optimizer.state_dict(),
            opt=opt,
            n_iter=n_iter,
            finetune_dict=dataloader.finetune_dict,
        ), opt.model_file + f'_finetuned_on_simulations.model')
        if (n_iter / opt.epoch_size) % 10 == 0:
            torch.save(dict(
                model=model,
                optimizer=optimizer.state_dict(),
                opt=opt,
                n_iter=n_iter,
                finetune_dict=dataloader.finetune_dict,
            ), opt.model_file + + f'_finetuned_on_simulations_step{n_iter}.model')


        model.to(opt.device)

        log_string = f'step {n_iter} | '
        log_string += 'train: [' + ', '.join(f'{k}: {train_losses[v]:.4f}' for k, v in losses.items()) + '] | '
        log_string += 'valid: [' + ', '.join(f'{k}: {valid_losses[v]:.4f}' for k, v in losses.items()) + ']'
#         print(log_string)
        utils.log(opt.model_file + f'_{date_str}.log', log_string)

        if valid_losses['policy'] < best_loss:
            best_loss = valid_losses['policy']
            patience = 0
        else:
            patience += 1
            
        if patience >= opt.finetune_earlystop_patience:
            log_string = f'early stopping at {i}\'th epoch'
            utils.log(opt.model_file + f'_{date_str}.log', log_string)
            break
            
        if writer is not None:
            writer.close()
    
#     print(train_eval_losses)


