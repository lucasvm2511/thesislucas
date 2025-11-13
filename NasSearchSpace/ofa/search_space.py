import numpy as np

def getOFASearchSpace(supernet, lr, ur, rstep):
    if 'mobilenetv3' in supernet:
        print('ja')
        return OFAMobileNetV3SearchSpace(supernet,lr, ur, rstep)
    elif 'resnet50' in supernet:
        return OFAResnet50SearchSpace(supernet,lr, ur, rstep)
    else:
        raise NotImplementedError

class OFAMobileNetV3SearchSpace:
    
    def __init__(self,supernet,lr,ur,rstep):

        self.num_blocks = 5  # number of blocks, default 5
        self.supernet = supernet

        if(supernet == 'mobilenetv3'):
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.nvar= 45 + int(lr!=ur) # length of the encoding 45 if fix_res else 46
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition 
        elif(supernet == 'eemobilenetv3'): # Early Exit Mbv3 (EDANAS)
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
            self.threshold = [0.1, 0.2, 1] #threshold value for selection scheme
            self.nvar=49 + int(lr!=ur)
        elif(supernet == 'cbnmobilenetv3'): #NACHOS
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
            self.num_branches=4
            self.branches = [0,1] #0=EEC, 1=No EEC
            self.nvar=49 + int(lr!=ur)
        elif(supernet == 'skippingmobilenetv3'):
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.nvar= 45 + int(lr!=ur) # length of the encoding 45 if fix_res else 46
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition 
        elif(supernet == 'skippingmobilenetv3_extended'):
            # Existing architecture parameters
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
            
            # Gate parameters
            self.gate_hidden_sizes = [16, 32, 64]  # Hidden layer sizes
            self.target_sparsity_options = [0, 0.3, 0.5, 0.7]  # 0 = no gate, others = target sparsity
            
            # Max number of blocks that could have gates: sum(max_depth) = 20
            max_depth = self.depth[-1]  
            self.max_gatable_blocks = sum([max_depth] * self.num_blocks)
            
            # Updated encoding length: 45 (base) + 1 (gate_hidden_size) + max_gatable_blocks (target_sparsities) + resolution
            self.nvar = 45 + 1 + self.max_gatable_blocks + int(lr!=ur)
        else:
            raise NotImplementedError

        #STANDARD is lr = 192 and ur= 256
        min = lr
        max = ur + 1
        self.resolution = list(range(min, max, rstep))
        self.fix_res = lr==ur

    def sample(self, n_samples=1, nb=None, ks=None, e=None, d=None, t = None, r=None, b=None):

        """ randomly sample a architecture"""
        nb = self.num_blocks if nb is None else nb
        ks = self.kernel_size if ks is None else ks
        e = self.exp_ratio if e is None else e
        d = self.depth if d is None else d
        if (self.supernet == 'eemobilenetv3'):
          t = self.threshold if t is None else t
        r = self.resolution if r is None else r
        if (self.supernet == 'cbnmobilenetv3'):
          #ne = self.ne if ne is None else ne
          b = self.branches if b is None else b
    
        data = []
        for n in range(n_samples):
            # first sample layers
            depth = np.random.choice(d, nb, replace=True).tolist()

            # then sample kernel size, expansion rate and resolution
            if(self.supernet == 'resnet50_he'):
              kernel_size = np.random.choice(ks, size=len(depth), replace=True).tolist()
              exp_ratio = np.random.choice(e, size=len(depth), replace=True).tolist()
            else:
              kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
              exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()

            if (self.supernet == 'eemobilenetv3'):

                threshold = np.random.choice(t, size=(len(depth)-1), replace=True).tolist()
                if self.fix_res:
                    data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 't': threshold})
                else:
                    resolution = int(np.random.choice(r))
                    data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 't': threshold, 'r': resolution})

            elif (self.supernet == 'mobilenetv3' or self.supernet == 'resnet50' or self.supernet == 'resnet50_he' or self.supernet == 'skippingmobilenetv3'):

                if self.fix_res:
                    data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth})
                else:
                    resolution = int(np.random.choice(r))
                    data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r':resolution})

            elif (self.supernet == 'cbnmobilenetv3'):

                #avoid branches [0,0,0,0] = no EECs
                while True:
                    branches = np.random.choice(b, self.num_branches, replace=True).tolist()
                    if any(branch != 0 for branch in branches):
                        break

                if self.fix_res:
                    data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'b': branches}) 
                else:
                    resolution = int(np.random.choice(r))
                    data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'b': branches, 'r':resolution})

            elif (self.supernet == 'skippingmobilenetv3_extended'):

                # Sample gate parameters
                gate_hidden_size = int(np.random.choice(self.gate_hidden_sizes))
                
                # Calculate actual number of blocks for this architecture
                num_blocks_actual = int(np.sum(depth)) - 4  # Subtract 4 for edge blocks
                
                # Sample target sparsities for each potential gatable block (length = max_gatable_blocks)
                # 0 means no gate, others are sparsity targets
                target_sparsities = np.random.choice(
                    self.target_sparsity_options, 
                    size=self.max_gatable_blocks, 
                    replace=True
                ).tolist()

                if self.fix_res:
                    data.append({
                        'ks': kernel_size, 'e': exp_ratio, 'd': depth,
                        'gate_hidden_size': gate_hidden_size,
                        'target_sparsities': target_sparsities  # Array of length max_gatable_blocks
                    })
                else:
                    resolution = int(np.random.choice(r))
                    data.append({
                        'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r': resolution,
                        'gate_hidden_size': gate_hidden_size,
                        'target_sparsities': target_sparsities  # Array of length max_gatable_blocks
                    })

        return data

    def initialize(self, n_doe):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        if (self.supernet == 'mobilenetv3' or self.supernet == 'skippingmobilenetv3'):

            if self.fix_res:
                data = [
                self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                            d=[min(self.depth)])[0],
                self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                            d=[max(self.depth)])[0]]
            else:
                data = [
                    self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                                d=[min(self.depth)], r=[min(self.resolution)])[0],
                    self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                                d=[max(self.depth)], r=[max(self.resolution)])[0]
                ]

        elif (self.supernet == 'eemobilenetv3'):
            data = [
                self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                            d=[min(self.depth)], t = [min(self.threshold)], r=[min(self.resolution)])[0],
                self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                            d=[max(self.depth)], t = [max(self.threshold)], r=[max(self.resolution)])[0]
            ]
        elif (self.supernet == 'cbnmobilenetv3'):
            data = [
                self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                            d=[min(self.depth)], b=[min(self.branches)])[0], # all EECs
                self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                            d=[max(self.depth)], b=[max(self.branches)])[0]  # no EECs
            ]
        elif (self.supernet == 'skippingmobilenetv3_extended'):
            if self.fix_res:
                data = [
                    self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                                d=[min(self.depth)])[0],
                    self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                                d=[max(self.depth)])[0]
                ]
            else:
                data = [
                    self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                                d=[min(self.depth)], r=[min(self.resolution)])[0],
                    self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                                d=[max(self.depth)], r=[max(self.resolution)])[0]
                ]
        else:
            print("Not yet implemented!")

        data.extend(self.sample(n_samples=n_doe - 2))
        return data

    def pad_zero(self, x, depth):
        # pad zeros to make bit-string of equal length
        new_x, counter = [], 0
        for d in depth:
            for _ in range(d):
                new_x.append(x[counter])
                counter += 1
            if d < max(self.depth):
                new_x += [0] * (max(self.depth) - d)
        return new_x

    def encode(self, config):
        # encode config ({'ks': , 'd': , etc}) to integer bit-string [1, 0, 2, 1, ...]
        x = []
        depth = [np.argwhere(_x == np.array(self.depth))[0, 0] for _x in config['d']]
        kernel_size = [np.argwhere(_x == np.array(self.kernel_size))[0, 0] for _x in config['ks']]
        exp_ratio = [np.argwhere(_x == np.array(self.exp_ratio))[0, 0] for _x in config['e']]

        kernel_size = self.pad_zero(kernel_size, config['d'])
        exp_ratio = self.pad_zero(exp_ratio, config['d'])
        for i in range(len(depth)):
            x = x + [depth[i]] + kernel_size[i * max(self.depth):i * max(self.depth) + max(self.depth)] \
                + exp_ratio[i * max(self.depth):i * max(self.depth) + max(self.depth)]
        
        if (self.supernet == 'eemobilenetv3'):
            idxs = [np.argwhere(_x == np.array(self.threshold))[0, 0] for _x in config['t']]
            x = x + idxs

        if (self.supernet == 'cbnmobilenetv3'):
            branches = config['b']
            for i in range(self.num_branches):
                x = x + [np.argwhere(branches[i] == np.array(self.branches))[0, 0]]
        
        if (self.supernet == 'skippingmobilenetv3_extended'):
            # Encode the gate parameters
            gate_hidden_size_idx = np.argwhere(config['gate_hidden_size'] == np.array(self.gate_hidden_sizes))[0, 0]
            
            # Encode target_sparsities array (fixed length = max_gatable_blocks)
            target_sparsity_indices = [
                np.argwhere(ts == np.array(self.target_sparsity_options))[0, 0] 
                for ts in config['target_sparsities']
            ]
            
            x = x + [gate_hidden_size_idx] + target_sparsity_indices  # 1 + max_gatable_blocks elements
        
        if not self.fix_res:
                x.append(np.argwhere(config['r'] == np.array(self.resolution))[0, 0])

        return x

    def decode(self, x):
        """
        remove un-expressed part of the chromosome
        assumes x = [block1, block2, ..., block5, gate_params, resolution];
        block_i = [depth, kernel_size, exp_rate]
        """
        
        x = x.astype(int)

        depth, kernel_size, exp_rate = [], [], []
        step = 1 + 2 * max(self.depth)
        
        # Calculate where architecture encoding ends based on supernet type
        if (self.supernet == 'skippingmobilenetv3_extended'):
            # Architecture is first 45 elements (5 blocks * 9 elements each)
            arch_end = 45
        else:
            # For other supernets, use the old logic
            arch_end = len(x) - 5

        for i in range(0, arch_end, step):
            depth.append(self.depth[x[i]])
            kernel_size.extend(np.array(self.kernel_size)[x[i + 1:i + 1 + self.depth[x[i]]]].tolist())
            exp_rate.extend(np.array(self.exp_ratio)[x[i + 5:i + 5 + self.depth[x[i]]]].tolist())
        
        if (self.supernet == 'mobilenetv3' or self.supernet == 'skippingmobilenetv3'):

            if self.fix_res:
                return {'ks': kernel_size, 'e': exp_rate, 'd': depth}
            else: 
                return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 'r': self.resolution[x[-1]]}    

        elif (self.supernet == 'eemobilenetv3'): 
            t_config = x[-self.num_blocks:-1]
            t = []
            for c in t_config:
              t.append(self.threshold[c])   

            if not self.fix_res:
                return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 
                't': t, 'r': self.resolution[x[-1]]}
            else: # fixed resolution
                return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 't': t}

        elif (self.supernet == 'cbnmobilenetv3'):
            branches = []
            for i in range(len(x) - self.num_branches, len(x)):
              branches.append(self.branches[x[i]])
            return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 'b': branches}
        elif(self.supernet == 'skippingmobilenetv3_extended'):
            # Decode the gate parameters
            if self.fix_res:
                gate_params_start = -(1 + self.max_gatable_blocks)  # gate_hidden_size + target_sparsities
            else:
                gate_params_start = -(1 + self.max_gatable_blocks + 1)  # + resolution
            
            gate_hidden_size = self.gate_hidden_sizes[x[gate_params_start]]
            
            # Decode target_sparsities array (fixed length = max_gatable_blocks)
            target_sparsity_start = gate_params_start + 1
            # For negative indices, we need to handle carefully
            if gate_params_start < 0:
                # Convert negative index to positive for slicing
                target_sparsity_start_pos = len(x) + gate_params_start + 1
                target_sparsity_end_pos = target_sparsity_start_pos + self.max_gatable_blocks
                target_sparsity_indices = x[target_sparsity_start_pos:target_sparsity_end_pos]
            else:
                target_sparsity_end = target_sparsity_start + self.max_gatable_blocks
                target_sparsity_indices = x[target_sparsity_start:target_sparsity_end]
            
            target_sparsities = [self.target_sparsity_options[i] for i in target_sparsity_indices]
            
            if self.fix_res:
                return {
                    'ks': kernel_size, 'e': exp_rate, 'd': depth,
                    'gate_hidden_size': gate_hidden_size,
                    'target_sparsities': target_sparsities  # Array of floats (includes 0s)
                }
            else:
                return {
                    'ks': kernel_size, 'e': exp_rate, 'd': depth, 'r': self.resolution[x[-1]],
                    'gate_hidden_size': gate_hidden_size,
                    'target_sparsities': target_sparsities  # Array of floats (includes 0s)
                }
        else: 
            print("Not yet implemented!")


class OFAResnet50SearchSpace:
    
    def __init__(self, supernet,lr, ur, rstep):
        self.num_stages = 5  
        self.num_blocks = 18
        self.exp_ratio = [0.2, 0.25, 0.35]  # expansion rate (e)
        self.depth = [0, 1, 2]  # number of Inverted Residual Bottleneck layers repetition
        self.width_mult = [0, 1, 2]  # width indices to width multipliers 0.65, 0.8, 1
        min_res = lr
        max_res = ur + 1
        self.resolution = list(range(min_res, max_res, rstep))
        self.fix_res = lr == ur
        self.nvar = 29 + int(lr!=ur) # number of variables in the encoding

    def sample(self, n_samples=1, d=None, e=None, w=None, r=None):
        """Randomly sample a configuration."""
        nb = self.num_blocks
        ns = self.num_stages
        e = self.exp_ratio if e is None else e
        d = self.depth if d is None else d
        w = self.width_mult if w is None else w
        r = self.resolution if r is None else r
        
        data = []
        for _ in range(n_samples):
            depth = np.random.choice(d, ns, replace=True).tolist()  # Length of d = nb (5 blocks)
            exp_ratio = np.random.choice(e, size=nb, replace=True).tolist()  # 18 expansion ratios (length = 18)
            width_mult = np.random.choice(w, size=(ns+1), replace=True).tolist()  # Weight multiplier (length = 6)
            resolution = int(np.random.choice(r)) if not self.fix_res else None
            
            config = {'d': depth, 'e': exp_ratio, 'w': width_mult}
            if not self.fix_res:
                config['r'] = resolution
            
            data.append(config)

        return data
    
    def initialize(self, n_doe=0):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)

        if self.fix_res:
            data = [
            self.sample(1, w=[min(self.width_mult)], e=[min(self.exp_ratio)],
                        d=[min(self.depth)])[0],
            self.sample(1, w=[max(self.width_mult)], e=[max(self.exp_ratio)],
                        d=[max(self.depth)])[0]]
        else:
            data = [
                self.sample(1, w=[min(self.width_mult)], e=[min(self.exp_ratio)],
                            d=[min(self.depth)], r=[min(self.resolution)])[0],
                self.sample(1, w=[max(self.width_mult)], e=[max(self.exp_ratio)],
                            d=[max(self.depth)], r=[max(self.resolution)])[0]
            ]
        data.extend(self.sample(n_samples=n_doe - 2))
        return data

    def encode(self, config):
        """Encode config to integer bit-string [1, 0, 2, 1, ...]."""
        x = []
        depth = [np.argwhere(_x == np.array(self.depth))[0, 0] for _x in config['d']]
        exp_ratio = [np.argwhere(_x == np.array(self.exp_ratio))[0, 0] for _x in config['e']]
        width_mult = [np.argwhere(_x == np.array(self.width_mult))[0, 0] for _x in config['w']]

        x += depth + exp_ratio + width_mult

        if not self.fix_res:
            x.append(np.argwhere(config['r'] == np.array(self.resolution))[0, 0])

        return x

    def decode(self, x):
        """Decode integer bit-string to architecture configuration."""
        x = x.astype(int)
        depth = [self.depth[x[i]] for i in range(5)]  # First 5 elements for depth
        exp_ratio = [self.exp_ratio[x[i]] for i in range(5, 23)]  # Next 18 elements for expansion ratios
        width_mult = [self.width_mult[x[i]] for i in range(23, 29)]  # Next 6 elements for weight multipliers
        
        config = {'d': depth, 'e': exp_ratio, 'w': width_mult}
        
        if not self.fix_res:
            config['r'] = self.resolution[x[-1]]

        return config




