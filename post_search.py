import os
import json
import shutil
import argparse
import glob
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from explainability import get_archive
from utils import get_subnet_folder
#from ofa_evaluator import OFAEvaluator, get_net_info, get_adapt_net_info

from matplotlib import pyplot as plt


_DEBUG = False


def print_feature_analysis(archive, selected_indices, ps, pf=None, args=None):
    """
    Print comprehensive analysis of the most important features in selected architectures
    """
    print("\n" + "="*80)
    print("FEATURE ANALYSIS OF SELECTED ARCHITECTURES")
    print("="*80)
    
    # Extract all subnets and objectives
    all_subnets = [v[0] for v in archive]
    all_first_obj = [v[1] for v in archive]
    
    # Check if we have a second objective
    has_sec_obj = len(archive[0]) > 2
    if has_sec_obj:
        all_sec_obj = [v[2] for v in archive]
    
    # Analyze architecture configuration keys
    config_keys = set()
    for subnet in all_subnets:
        config_keys.update(subnet.keys())
    
    print(f"\n1. ARCHIVE OVERVIEW")
    print(f"   Total candidates in archive: {len(archive)}")
    print(f"   Selected architectures: {len(selected_indices)}")
    print(f"   Configuration keys: {sorted(config_keys)}")
    
    # Statistics for first objective
    print(f"\n2. FIRST OBJECTIVE ({args.first_obj if args else 'top1'}) STATISTICS")
    print(f"   Archive - Min: {np.min(all_first_obj):.4f}, Max: {np.max(all_first_obj):.4f}, Mean: {np.mean(all_first_obj):.4f}, Std: {np.std(all_first_obj):.4f}")
    selected_first_obj = [all_first_obj[i] for i in selected_indices]
    print(f"   Selected - Min: {np.min(selected_first_obj):.4f}, Max: {np.max(selected_first_obj):.4f}, Mean: {np.mean(selected_first_obj):.4f}")
    
    # Statistics for second objective if exists
    if has_sec_obj and args and args.sec_obj:
        print(f"\n3. SECOND OBJECTIVE ({args.sec_obj}) STATISTICS")
        print(f"   Archive - Min: {np.min(all_sec_obj):.4f}, Max: {np.max(all_sec_obj):.4f}, Mean: {np.mean(all_sec_obj):.4f}, Std: {np.std(all_sec_obj):.4f}")
        selected_sec_obj = [all_sec_obj[i] for i in selected_indices]
        print(f"   Selected - Min: {np.min(selected_sec_obj):.4f}, Max: {np.max(selected_sec_obj):.4f}, Mean: {np.mean(selected_sec_obj):.4f}")
    
    # Analyze architectural features
    print(f"\n4. ARCHITECTURAL FEATURES ANALYSIS")
    
    # Analyze specific configuration parameters
    for key in sorted(config_keys):
        print(f"\n   Feature: '{key}'")
        
        # Collect values for this key across all architectures
        all_values = []
        selected_values = []
        
        for i, subnet in enumerate(all_subnets):
            if key in subnet:
                val = subnet[key]
                all_values.append(val)
                if i in selected_indices:
                    selected_values.append(val)
        
        if not all_values:
            continue
            
        # Handle different types of values
        if isinstance(all_values[0], (list, tuple)):
            # For list/array features
            all_lengths = [len(v) for v in all_values]
            sel_lengths = [len(v) for v in selected_values]
            print(f"      Archive length - Min: {np.min(all_lengths)}, Max: {np.max(all_lengths)}, Mean: {np.mean(all_lengths):.2f}")
            if sel_lengths:
                print(f"      Selected length - Min: {np.min(sel_lengths)}, Max: {np.max(sel_lengths)}, Mean: {np.mean(sel_lengths):.2f}")
            
            # For exits configuration (b parameter)
            if key == 'b':
                all_exit_counts = [len([e for e in v if e != 0]) for v in all_values]
                sel_exit_counts = [len([e for e in v if e != 0]) for v in selected_values]
                print(f"      Archive exit count - Min: {np.min(all_exit_counts)}, Max: {np.max(all_exit_counts)}, Mean: {np.mean(all_exit_counts):.2f}")
                if sel_exit_counts:
                    print(f"      Selected exit count - Min: {np.min(sel_exit_counts)}, Max: {np.max(sel_exit_counts)}, Mean: {np.mean(sel_exit_counts):.2f}")
                    print(f"      Selected exit counts: {sel_exit_counts}")
                    
        elif isinstance(all_values[0], (int, float, np.number)):
            # For numeric features
            print(f"      Archive - Min: {np.min(all_values):.4f}, Max: {np.max(all_values):.4f}, Mean: {np.mean(all_values):.4f}")
            if selected_values:
                print(f"      Selected - Min: {np.min(selected_values):.4f}, Max: {np.max(selected_values):.4f}, Mean: {np.mean(selected_values):.4f}")
        else:
            # For other types, show unique values
            unique_all = set(str(v) for v in all_values)
            unique_sel = set(str(v) for v in selected_values)
            print(f"      Archive unique values: {len(unique_all)}")
            if unique_sel:
                print(f"      Selected unique values: {len(unique_sel)}")
    
    # Trade-off analysis if Pareto front exists
    if pf is not None and len(pf) > 0:
        print(f"\n5. PARETO FRONT ANALYSIS")
        print(f"   Number of Pareto optimal solutions: {len(pf)}")
        print(f"   Objective 1 range: [{pf[:, 0].min():.4f}, {pf[:, 0].max():.4f}]")
        print(f"   Objective 2 range: [{pf[:, 1].min():.4f}, {pf[:, 1].max():.4f}]")
        
        # Calculate hypervolume approximation (area covered)
        if len(pf) > 1:
            sorted_pf = pf[np.argsort(pf[:, 0])]
            area = 0
            for i in range(len(sorted_pf) - 1):
                width = sorted_pf[i+1, 0] - sorted_pf[i, 0]
                height = sorted_pf[i, 1]
                area += width * height
            print(f"   Approximate hypervolume: {area:.4f}")
    
    # Print detailed info for each selected architecture
    print(f"\n6. SELECTED ARCHITECTURES DETAILS")
    for rank, idx in enumerate(selected_indices):
        print(f"\n   Rank {rank} (Index {idx}):")
        print(f"      First objective: {all_first_obj[idx]:.4f}")
        if has_sec_obj:
            print(f"      Second objective: {all_sec_obj[idx]:.4f}")
        print(f"      Configuration: {ps[rank] if rank < len(ps) else all_subnets[idx]}")
    
    print("\n" + "="*80)
    print("END OF FEATURE ANALYSIS")
    print("="*80 + "\n")

                    
class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        #if self.normalize:
        #    F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            #np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def main(args):

    exp_path,_= os.path.splitext(args.expr)
    
    if args.get_archive:
       archive = get_archive(exp_path, args.first_obj, args.sec_obj)
    else:
       archive = json.load(open(args.expr))['archive']

    n_exits = args.n_exits
    if n_exits is not None:
        # filter according to n° of exits
        archive_temp = []
        for v in archive:
            subnet = v[0]
            b_config = subnet["b"]
            count_exits = len([element for element in b_config if element != 0])
            if(count_exits==args.n_exits):
                archive_temp.append(v)
        print("#EEcs:")
        print(args.n_exits)
        print("lunghezza archivio prima")        
        print(len(archive))
        archive = archive_temp
    
    print("NUM CANDIDATES")
    print(len(archive))

    if args.sec_obj is None:
        subnets, first_obj = [v[0] for v in archive], [v[1] for v in archive]
        prefer = args.first_obj
    else:
        subnets, first_obj, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
        prefer = 'trade-off'
        ps_sec_obj = np.array(sec_obj)

    if args.sec_obj is None:
        ps = np.array(subnets)
        ps_first_obj = np.array(first_obj)
        I = ps_first_obj.argsort()[:args.n]
        pf = None
    else:
        sort_idx = np.argsort(first_obj)
        F = np.column_stack((first_obj, sec_obj))[sort_idx, :]
        F = F[(F[:, 0] >= 0) & (F[:, 1] >= 0)] #remove negative values
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        ps = np.array(subnets)[sort_idx][front]
        pf = F[front, :]
        print(f"\nPareto front size: {len(pf)}")
        
        # Select diverse architectures using greedy diversity selection
        if len(pf) <= args.n:
            # If Pareto front is smaller than requested, take all
            I = np.arange(len(pf))
        else:
            # Normalize objectives to same scale for distance calculation
            pf_norm = (pf - pf.min(axis=0)) / (pf.max(axis=0) - pf.min(axis=0) + 1e-8)
            
            # Greedy selection: pick points that maximize minimum distance to already selected
            I = []
            # Start with the extreme points (best in each objective)
            I.append(np.argmin(pf[:, 0]))  # Best first objective
            if args.n > 1:
                I.append(np.argmin(pf[:, 1]))  # Best second objective
            
            # Add remaining points by maximizing minimum distance
            while len(I) < args.n:
                max_min_dist = -1
                best_idx = -1
                for candidate in range(len(pf)):
                    if candidate in I:
                        continue
                    # Calculate minimum distance to already selected points
                    min_dist = min([np.linalg.norm(pf_norm[candidate] - pf_norm[selected]) 
                                   for selected in I])
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = candidate
                if best_idx != -1:
                    I.append(best_idx)
                else:
                    break
            
            I = np.array(I)
        
        print(f"Selected {len(I)} diverse architectures spanning the Pareto front (requested: {args.n})")

    # Print feature analysis
    print_feature_analysis(archive, I, ps, pf=pf, args=args)

    # Deduplicate selected architectures based on configuration
    unique_configs = []
    unique_indices = []
    seen_configs = set()
    seen_performance = set()  # Track exact performance duplicates only
    
    print(f"\n{'='*60}")
    print(f"DEDUPLICATION PROCESS")
    print(f"{'='*60}")
    for i, idx in enumerate(I):
        config = ps[idx]
        config_str = json.dumps(config, sort_keys=True)
        
        # Create performance signature with more precision to only catch exact duplicates
        perf_signature = (round(pf[idx, 0], 6), round(pf[idx, 1], 6))
        
        is_config_dup = config_str in seen_configs
        is_perf_dup = perf_signature in seen_performance
        
        if not is_config_dup and not is_perf_dup:
            seen_configs.add(config_str)
            seen_performance.add(perf_signature)
            unique_configs.append(config)
            unique_indices.append(idx)
            print(f"Architecture {i} (idx={idx}): UNIQUE - obj1={pf[idx,0]:.4f}, obj2={pf[idx,1]:.4f} (total unique: {len(unique_configs)})")
        else:
            dup_type = "config" if is_config_dup else "performance"
            print(f"Architecture {i} (idx={idx}): DUPLICATE ({dup_type}) - obj1={pf[idx,0]:.4f}, obj2={pf[idx,1]:.4f} - skipping")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Selected {len(I)} architectures, {len(unique_configs)} unique after deduplication")
    print(f"{'='*60}\n")
    
    if len(unique_configs) == 0:
        print("ERROR: No unique configurations found! Exiting.")
        return
    
    # always add most accurate architectures
    #I = np.append(I, 0)

    # create the supernet
    #supernet = OFAEvaluator(n_classes = args.n_classes, model_path=args.supernet_path, pretrained = args.supernet_path)

    for rank, (idx, config) in enumerate(zip(unique_indices, unique_configs)):

        if(n_exits is not None):
          save = os.path.join(args.save, "net-"+ prefer +"_"+str(rank)+"_nExits:"+str(args.n_exits))
        else:
          save = os.path.join(args.save, "net-"+ prefer +"_"+str(rank))

        os.makedirs(save, exist_ok=True)
        print("CONFIG {}: {}".format(rank, config))

        #subnet, _ = supernet.sample(config)
        subnet_folder = get_subnet_folder(exp_path, config)
        shutil.rmtree(save, ignore_errors=True)
        shutil.copytree(subnet_folder, save)
        #n_subnet = subnet_folder.rsplit("_", 1)[1]
        subnet_file = [filename for filename in os.listdir(save) if filename.endswith('.subnet')][0]
        stats_file = [filename for filename in os.listdir(save) if filename.endswith('.stats')][0]
        os.rename(os.path.join(save, subnet_file), os.path.join(save, "net.subnet"))
        os.rename(os.path.join(save, stats_file), os.path.join(save, "net.stats"))   

        print("SUBNET FOLDER: {}".format(subnet_folder))    

        stats_file = os.path.join(save, "net.stats")
        
        if os.path.exists(stats_file):
            
            stats = json.load(open(stats_file))
            print("INFO SUBNET RANK {}".format(rank))
            print(stats)

    if _DEBUG:
        # Plot

        pf = np.array(pf)
        x = pf[:,0]
        y = pf[:,1]
        plt.scatter(x, y, c='red')

        plt.title('Pareto front')
        plt.xlabel('1-top1')
        plt.ylabel('sec_obj')
        plt.legend()
        plt.show()
        plt.savefig(args.save + 'scatter_plot_pareto_front.png')

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--expr', type=str, default='',
                        help='location of search experiment dir')
    parser.add_argument('--first_obj', type=str, default='top1',
                        help='second objective to optimize')
    parser.add_argument('--sec_obj', type=str, default=None,
                        help='second objective to optimize')
    parser.add_argument('--n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--search_space', type=str, default='mobilenetv3',
                        help='type of search space')
    parser.add_argument('--get_archive', action='store_true', default=False,
                        help='create the archive scanning the iter folders')
    parser.add_argument('--n_classes', type=int, default=1000,
                        help='number of classes')                   
    parser.add_argument('--pmax', type = float, default=2.0,
                        help='max value of params for candidate architecture')
    parser.add_argument('--fmax', type = float, default=100,
                        help='max value of flops for candidate architecture')
    parser.add_argument('--amax', type = float, default=5.0,
                        help='max value of activations for candidate architecture')
    parser.add_argument('--wp', type = float, default=1.0,
                        help='weight for params')
    parser.add_argument('--wf', type = float, default=1/40,
                        help='weight for flops')
    parser.add_argument('--wa', type = float, default=1.0,
                        help='weight for activations')
    parser.add_argument('--penalty', type = float, default=10**10,
                        help='penalty factor')
    parser.add_argument('--n_exits', type=int, default=None,
                        help='number of EEcs desired')
    parser.add_argument('--lr', type = int , default=192,
                        help='minimum resolution')
    parser.add_argument('--ur', type = int, default=256,
                        help='maximum resolution')
    parser.add_argument('--rstep', type = int, default=4,
                        help='resolution step')
    cfgs = parser.parse_args()
    main(cfgs)

