import sys
import random, pdb, math, pickle, glob, time, os, re
import torch
import numpy as np
from natsort import natsorted


class DataLoader:
    def __init__(self, fname, opt, dataset='simulator', single_shard=False, all_batches=False, randomize=True):
        if opt.debug:
            single_shard = True
        self.opt = opt
        self.random = random.Random()
        self.random.seed(12345)  # use this so that the same batches will always be picked

        if dataset == 'i80' or dataset == 'us101':
            data_dir = f'traffic-data/state-action-cost/data_{dataset}_v0'
        else:
            data_dir = dataset

        if single_shard:
            # quick load for debugging
            data_files = [f'{next(os.walk(data_dir))[1][0]}.txt/']
        else:
            data_files = next(os.walk(data_dir))[1]

        self.images = []
        self.actions = []
        self.costs = []
        self.states = []
    
        self.ids = []
        self.ego_car_images = []
        
#         if opt.training_method == 'finetune_sim':
#             costs_path = data_dir + ".costs"
#             costs = torch.load(costs_path)
#             self.costs = costs
#             print("costs: ",len(costs), len(costs[0]), costs[0][0].keys())

#             states_path = data_dir + ".states"
#             states = torch.load(states_path)
#             self.states = states
#             print("states: ",len(states), states[0].shape)

#             actions_path = data_dir + ".actions"
#             actions = torch.load(actions_path)
#             self.actions = actions
#             print("actions: ",len(actions), actions[0].shape)
            
#             others_path = data_dir + ".others"
#             others = torch.load(others_path)
#             road_completed = others["road_completed"]
#             collided = others["collided"]
#             offscreen = others["offscreen"]
#             self.car_sizes = others["car_sizes"]
            
#             path = data_dir.split("/")
#             model_path = path[-1]
#             path = path[:-1]
#             path += ["videos_simulator", model_path]
#             images_path = "/".join(path)

#             episode_files = natsorted(os.listdir(images_path))
# #             print(episode_files[:5])

#             episode_images = []
#             episode_ego_images = []
#             for episode in episode_files:

#                 images = natsorted(os.listdir(images_path + "/" + episode + "/all/"))
#                 if '.ipynb_checkpoints' in images:
#                     images.remove('.ipynb_checkpoints')
#                 for image in images:
#                     image = Image.open(images_path + "/" + episode + "/all/" + image)
# #                     print(np.array(image).shape)
#                     image = np.array(image)
                    
#                     episode_images.append(image)
                    
#                 self.images.append(episode_images)
                
#                 ego_images = natsorted(os.listdir(images_path + "/" + episode + "/ego/"))
#                 if '.ipynb_checkpoints' in ego_images:
#                     ego_images.remove('.ipynb_checkpoints')
#                 for ego_image in ego_images:
#                     ego_image = Image.open(images_path + "/" + episode + "/all/" + ego_image)
# #                     print(np.array(image).shape)
#                     ego_image = np.array(ego_image)
                    
#                     episode_ego_images.append(ego_image)
                    
#                 self.ego_car_images.append(episode_ego_images)
                
#             self.finetune_indx = []
#             for episode_i in range(len(self.images)):
#                 if not road_completed[episode_i]:
#                     self.states[episode_i] = self.states[episode_i][-opt.npred + -opt.ncond:]
#                     self.actions[episode_i] = self.actions[episode_i][-opt.npred + -opt.ncond:]
#                     self.costs[episode_i] = self.costs[episode_i][-opt.npred + -opt.ncond:]
#                     self.images[episode_i] = self.images[episode_i][-opt.npred + -opt.ncond:]
#                     self.ego_car_images[episode_i] = self.ego_car_images[episode_i][-opt.npred + -opt.ncond:]
#                     self.finetune_indx += episode_i
                    
#             combined_data_path = f'{data_dir}/all_data.pth'
#             torch.save({
#                     'images': self.images,
#                     'actions': self.actions,
#                     'costs': self.costs,
#                     'states': self.states,
#                     'ids': self.ids,
#                     'ego_car': self.ego_car_images,
#                 }, combined_data_path)
            
#             self.n_episodes = len(self.images)
#             print(f'Number of episodes: {self.n_episodes}')
            
            
        
#             splits_path = f'traffic-data/state-action-cost/data_{dataset}_v0' + '/splits.pth'
        
                
#             car_sizes_path = f'traffic-data/state-action-cost/data_{dataset}_v0' + '/car_sizes.pth'

#         else:
        
        for df in data_files:
            combined_data_path = f'{data_dir}/{df}/all_data.pth'
            if os.path.isfile(combined_data_path):
                print(f'[loading data shard: {combined_data_path}]')
                data = torch.load(combined_data_path)
                self.images += data.get('images')
                self.actions += data.get('actions')
                self.costs += data.get('costs')
                self.states += data.get('states')
                self.ids += data.get('ids')
                self.ego_car_images += data.get('ego_car')
#                 print(len(data.get('images')))
                data_images = data.get('images')

#                 print(type(data_images),len(data_images),type(data_images[0]),len(data_images[0]),data_images[0].shape)
            else:
                print(data_dir)
                images = []
                actions = []
                costs = []
                states = []
                ids = glob.glob(f'{data_dir}/{df}/car*.pkl')
                ids.sort()
                ego_car_images = []
                for f in ids:
                    print(f'[loading {f}]')
                    fd = pickle.load(open(f, 'rb'))
                    Ta = fd['actions'].size(0)
                    Tp = fd['pixel_proximity_cost'].size(0)
                    Tl = fd['lane_cost'].size(0)
                    # assert Ta == Tp == Tl  # TODO Check why there are more costs than actions
                    # if not(Ta == Tp == Tl): pdb.set_trace()
                    images.append(fd['images'])
                    actions.append(fd['actions'])
                    costs.append(torch.cat((
                        fd.get('pixel_proximity_cost')[:Ta].view(-1, 1),
                        fd.get('lane_cost')[:Ta].view(-1, 1),
                    ), 1),)
                    states.append(fd['states'])
                    ego_car_images.append(fd['ego_car'])

                print(f'Saving {combined_data_path} to disk')
                torch.save({
                    'images': images,
                    'actions': actions,
                    'costs': costs,
                    'states': states,
                    'ids': ids,
                    'ego_car': ego_car_images,
                }, combined_data_path)
                self.images += images
                self.actions += actions
                self.costs += costs
                self.states += states
                self.ids += ids
                self.ego_car_images += ego_car_images

        self.n_episodes = len(self.images)
        print(f'Number of episodes: {self.n_episodes}')

        f'traffic-data/state-action-cost/data_{dataset}_v0'

        splits_path = data_dir + '/splits.pth'
        
                
        car_sizes_path = data_dir + '/car_sizes.pth'
        print(f'[loading car sizes: {car_sizes_path}]')
        self.car_sizes = torch.load(car_sizes_path)
            
        if os.path.exists(splits_path):
            print(f'[loading data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.train_indx = self.splits.get('train_indx')
            self.valid_indx = self.splits.get('valid_indx')
            self.test_indx = self.splits.get('test_indx')
        else:
            print('[generating data splits]')
            rgn = np.random.RandomState(0)
            perm = rgn.permutation(self.n_episodes)
            n_train = int(math.floor(self.n_episodes * 0.8))
            n_valid = int(math.floor(self.n_episodes * 0.1))
            self.train_indx = perm[0 : n_train]
            self.valid_indx = perm[n_train : n_train + n_valid]
            self.test_indx = perm[n_train + n_valid :]
            torch.save(dict(
                train_indx=self.train_indx,
                valid_indx=self.valid_indx,
                test_indx=self.test_indx,
            ), splits_path)     
        
        stats_path = data_dir + '/data_stats.pth'
        if os.path.isfile(stats_path):
            print(f'[loading data stats: {stats_path}]')
            stats = torch.load(stats_path)
            self.a_mean = stats.get('a_mean')
            self.a_std = stats.get('a_std')
            self.s_mean = stats.get('s_mean')
            self.s_std = stats.get('s_std')
        else:
            print('[computing action stats]')
            all_actions = []
            for i in self.train_indx:
                all_actions.append(self.actions[i])
            all_actions = torch.cat(all_actions, 0)
            self.a_mean = torch.mean(all_actions, 0)
            self.a_std = torch.std(all_actions, 0)
            print('[computing state stats]')
            all_states = []
            for i in self.train_indx:
                all_states.append(self.states[i][:, 0])
            all_states = torch.cat(all_states, 0)
            self.s_mean = torch.mean(all_states, 0)
            self.s_std = torch.std(all_states, 0)
            torch.save({'a_mean': self.a_mean,
                        'a_std': self.a_std,
                        's_mean': self.s_mean,
                        's_std': self.s_std}, stats_path)
        
        self.what = None
        if all_batches:
            self.total_instances = {}
            self.episode_index = 0
            self.timestep = 0
            self.episode_dict = {}
            if opt.training_method == 'finetune_train':
                self.finetune_dict = {}
                self.t_finetune_dict = None
            
    def get_total_instances(self, split, what):
        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx
        else:# split == 'finetune_train':
            indx = list(self.finetune_dict.keys())
#         elif split == 'finetune_sim':
#             indx = self.finetune_indx
            
        self.total_instances[split] = 0
        
#         if what == 'finetune' and self.opt.finetune_nepisodes > 0:
#             indx = indx[:self.opt.finetune_nepisodes]
        if what == 'eval' and self.opt.eval_nepisodes > 0:
            indx = indx[:self.opt.eval_nepisodes]
            
        self.what = what
        overlap = self.opt.eval_nframes_overlap if what == 'eval' else self.opt.finetune_nframes_overlap
        
        print("len of indx in get total instances:", len(indx))
        for index in indx:
#             print("index:",index)
#             print("images[index]:",self.images[index].shape)
#             print(min(self.images[index].size(0), self.states[index].size(0)),(self.opt.ncond + self.opt.npred),overlap)
#             print((min(self.images[index].size(0), self.states[index].size(0)) - (self.opt.ncond + self.opt.npred)),(min(self.images[index].size(0), self.states[index].size(0)) - (self.opt.ncond + self.opt.npred))//overlap)
            if split == 'finetune_train':
                self.total_instances[split] += len(self.finetune_dict[index])
#             elif split == 'finetune_sim':
#                 self.total_instances[split] += len(self.finetune_indx[index])
            else:
                self.total_instances[split] += (min(self.images[index].size(0), self.states[index].size(0)) - (self.opt.ncond + self.opt.npred))//overlap + 1
        
        
        return self.total_instances[split]
        
        
    def set_indices_dict(self, split):
        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx
        print(len(indx))
            
        if split not in self.episode_dict:
            self.episode_dict[split] = {}
            
        if self.opt.finetune_nepisodes>0:
            indx = indx[:self.opt.finetune_nepisodes]
        
        for index in indx:
            episode_length = min(self.images[index].size(0), self.states[index].size(0))
            self.episode_dict[split][index] = np.arange(episode_length - (self.opt.npred + self.opt.ncond))
                
    
    # get batch to use for forward modeling
    # a sequence of ncond given states, a sequence of npred actions,
    # and a sequence of npred states to be predicted
    def get_batch_fm(self, split, npred=-1, cuda=True, return_episode_index = False,  all_batches = False, randomize = True, what='train'):

        # Choose the correct device
        flag = 0
        device = torch.device('cuda') if cuda else torch.device('cpu')

        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx
        elif split == 'finetune_train':
#             print(self.t_finetune_dict)
            if not self.t_finetune_dict:
                self.t_finetune_dict = self.finetune_dict.copy()
            indx = list(self.t_finetune_dict.keys())
        elif split == 'finetune_sim':
            sim = self.finetune_indx
        
#         print(len(indx))
        if self.what and self.what == 'eval' and self.opt.eval_nepisodes > 0:
            indx = indx[:self.opt.eval_nepisodes]
        
        if all_batches:
            
            if not randomize and self.episode_index is None:
                self.episode_index = 0
                self.timestep = 0

        if npred == -1:
            npred = self.opt.npred

#         print("train ind ex, len(self.train_indx)")
#         print(indx, len(indx), self.total_instances)
        images, states, actions, costs, ids, sizes, ego_cars = [], [], [], [], [], [], []
        nb = 0
        if all_batches:
            episodes_indices = np.zeros((self.opt.batch_size,2)).astype(int)
        
        T = self.opt.ncond + npred
        while nb < self.opt.batch_size:
            if randomize:
#                 print("indx: ",indx)
                if all_batches:
#                     if split == 'finetune_train':
                    if not self.t_finetune_dict:
                        break
                    else:
                        s = self.random.choice(list(self.t_finetune_dict.keys()))
#                     elif split == 'finetune_sim':
#                         s = self.random.choice(indx)
                else:
                    s = self.random.choice(indx)
            else:
#                 s = self.random.choice(indx)
                if self.episode_index is None:
                    break
                s = indx[self.episode_index]
            # min is important since sometimes numbers do not align causing issues in stack operation below
            episode_length = min(self.images[s].size(0), self.states[s].size(0))
            if episode_length >= T:
                if randomize:
#                     s = self.random.choice(indx)
                    if all_batches:
#                         print("only s:",s)
#                         print("s and len of dict[s]",s,len(self.t_finetune_dict[s]))
                        t = self.random.choice(self.t_finetune_dict[s])
                        index, = np.where(self.t_finetune_dict[s]==t)
                        self.t_finetune_dict[s] = np.delete(self.t_finetune_dict[s],index)
                        
#                         print("s,t,len(dict[s])",s,t,len(self.t_finetune_dict[s]))
                        if len(self.t_finetune_dict[s])==0:
#                             print(f"Done with episode no: {s}")
                            del self.t_finetune_dict[s]
                            if len(list(self.t_finetune_dict)) == 0:
                                self.t_finetune_dict = None
#                             print("after deletion:",self.t_finetune_dict)
                            
                        
#                         print(len(images), self.total_instances[split], len(self.episode_dict[split]))
                    else:
                        t = self.random.randint(0, episode_length - T)
                else:
#                     s = self.episode_index
                    t = self.timestep
                
#                 print(s,t,len(self.images),episode_length, len(self.images[s]), len(self.states[s]), self.images[s][t : t + T].shape, self.states[s][t : t + T, 0].shape)
                images.append(self.images[s][t : t + T].to(device))
                actions.append(self.actions[s][t : t + T].to(device))
                states.append(self.states[s][t : t + T, 0].to(device))  # discard 6 neighbouring cars
                costs.append(self.costs[s][t : t + T].to(device))
#                 if split == 'finetune_sim':
#                     ids.append(self.ids[self.test_indx[s]])
#                 else:
                ids.append(self.ids[s])
                
                ego_cars.append(self.ego_car_images[s].to(device))
                
#                 if split == 'finetune_sim':
#                     splits = self.ids[self.test_indx[s]].split('/')
#                 else:
                splits = self.ids[s].split('/')
                    
                time_slot = splits[-2]
                car_id = int(re.findall(r'car(\d+).pkl', splits[-1])[0])
#                 if split == 'finetune_sim':
#                     size = self.car_sizes[s]
#                 else:
                size = self.car_sizes[time_slot][car_id]
                sizes.append([size[0], size[1]])
                if all_batches:
                    episodes_indices[nb] = np.array([s,t])
                    
                if not randomize:
#                     print(self.episode_index,self.timestep,len(self.images[s]),self.total_instances[split],len(indx),self.opt.eval_nepisodes)
                    self.timestep+=self.opt.eval_nframes_overlap
                    if self.timestep > episode_length - T:
                        self.timestep = 0
                        self.episode_index += 1
                        print(f"Done with {self.episode_index} episodes")
#                         print(type(self.episode_index), type(indx))
                        if (self.opt.eval_nepisodes > 0 and self.episode_index == self.opt.eval_nepisodes) or self.episode_index == len(indx):#len(indx):
                            self.episode_index = None
                            break
                
                nb += 1

        # Pile up stuff
#         print(len(images), self.total_instances[split])
        images  = torch.stack(images)
        states  = torch.stack(states)
        actions = torch.stack(actions)
        sizes   = torch.tensor(sizes)
        ego_cars = torch.stack(ego_cars)

        # Normalise actions, state_vectors, state_images
        if not self.opt.debug:
            actions = self.normalise_action(actions)
            states = self.normalise_state_vector(states)
        images = self.normalise_state_image(images)
        ego_cars = self.normalise_state_image(ego_cars)

        costs = torch.stack(costs)

        # |-----ncond-----||------------npred------------||
        # ^                ^                              ^
        # 0               t0                             t1
        t0 = self.opt.ncond
        t1 = T
        input_images  = images [:,   :t0].float().contiguous()
        input_states  = states [:,   :t0].float().contiguous()
        target_images = images [:, t0:t1].float().contiguous()
        target_states = states [:, t0:t1].float().contiguous()
        target_costs  = costs  [:, t0:t1].float().contiguous()
        t0 -= 1; t1 -= 1
        actions       = actions[:, t0:t1].float().contiguous()
        # input_actions = actions[:, :t0].float().contiguous()
        ego_cars = ego_cars.float().contiguous()
        #          n_cond                      n_pred
        # <---------------------><---------------------------------->
        # .                     ..                                  .
        # +---------------------+.                                  .  ^          ^
        # |i|i|i|i|i|i|i|i|i|i|i|.  3 × 117 × 24                    .  |          |
        # +---------------------+.                                  .  | inputs   |
        # +---------------------+.                                  .  |          |
        # |s|s|s|s|s|s|s|s|s|s|s|.  4                               .  |          |
        # +---------------------+.                                  .  v          |
        # .                   +-----------------------------------+ .  ^          |
        # .                2  |a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a| .  | actions  |
        # .                   +-----------------------------------+ .  v          |
        # .                     +-----------------------------------+  ^          | tensors
        # .       3 × 117 × 24  |i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|  |          |
        # .                     +-----------------------------------+  |          |
        # .                     +-----------------------------------+  |          |
        # .                  4  |s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|  | targets  |
        # .                     +-----------------------------------+  |          |
        # .                     +-----------------------------------+  |          |
        # .                  2  |c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|  |          |
        # .                     +-----------------------------------+  v          v
        # +---------------------------------------------------------+             ^
        # |                           car_id                        |             | string
        # +---------------------------------------------------------+             v
        # +---------------------------------------------------------+             ^
        # |                          car_size                       |  2          | tensor
        # +---------------------------------------------------------+             v

        if return_episode_index:
            return episodes_indices, [input_images, input_states, ego_cars], actions, [target_images, target_states, target_costs], ids, sizes
        
        return [input_images, input_states, ego_cars], actions, [target_images, target_states, target_costs], ids, sizes

    @staticmethod
    def normalise_state_image(images):
        return images.float().div_(255.0)

    def normalise_state_vector(self, states):
        shape = (1, 1, 4) if states.dim() == 3 else (1, 4)  # dim = 3: state sequence, dim = 2: single state
        states -= self.s_mean.view(*shape).expand(states.size()).to(states.device)
        states /= (1e-8 + self.s_std.view(*shape).expand(states.size())).to(states.device)
        return states

    def normalise_action(self, actions):
        actions -= self.a_mean.view(1, 1, 2).expand(actions.size()).to(actions.device)
        actions /= (1e-8 + self.a_std.view(1, 1, 2).expand(actions.size())).to(actions.device)
        return actions


if __name__ == '__main__':
    # Create some dummy options
    class DataSettings:
        debug = False
        batch_size = 4
        npred = 20
        ncond = 10
    # Instantiate data set object
    d = DataLoader(None, opt=DataSettings, dataset='i80')
    # Retrieve first training batch
    x = d.get_batch_fm('train', cuda=False)
