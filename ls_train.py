import json
import logging
import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import gc

import sys
sys.path.append(os.getcwd())

from train_utils import get_data_loaders, get_optimizer, get_loss, get_lr_scheduler, initialize_seed, train, validate, load_checkpoint, Log
from utils import get_network_search
from LayerSkipping.utils_ls import get_skipping_mobilenetv3

#--trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000
#init_lr=0.01, lr_schedule_type='cosine' weight_decay=4e-5, label_smoothing=0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mobilenetv3', help='name of the model (mobilenetv3, ...)')
    parser.add_argument('--ofa', action='store_true', default=True, help='s')
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="0.01 Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--n_workers", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="L2 weight decay.")
    parser.add_argument('--val_split', default=0.0, type=float, help='use validation set')
    parser.add_argument('--optim', type=str, default='SGD', help='algorithm to use for training')
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument('--dataset', type=str, default='imagenet', help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument("--data_aug", default=True, type=bool, help="True if you want to use data augmentation.")
    parser.add_argument('--save', action='store_true', default=False, help='save checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for training / testing')
    parser.add_argument('--n_classes', type=int, default=100, help='number of classes of the given dataset')
    parser.add_argument('--supernet_path', type=str, default='./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0', help='file path to supernet weights')
    parser.add_argument('--model_path', type=str, default=None, help='file path to subnet')
    parser.add_argument('--output_path', type=str, default=None, help='file path to save results')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained weights')
    parser.add_argument('--mmax', type=float, default=1000, help='maximum number of MACS allowed')
    parser.add_argument('--top1min', type=float, default=0.0, help='minimum top1 accuracy allowed')
    parser.add_argument("--use_early_stopping", default=True, type=bool, help="True if you want to use early stopping.")
    parser.add_argument("--early_stopping_tolerance", default=5, type=int, help="Number of epochs to wait before early stopping.")
    parser.add_argument("--resolution", default=32, type=int, help="Image resolution.")
    parser.add_argument("--func_constr", action='store_true', default=False, help='use functional constraints')

    # Training parameters
    parser.add_argument('--eval_test', action='store_true', default=True, help='evaluate test accuracy')
    parser.add_argument("--backbone_epochs", default=0, type=int, help="Number of epochs to train the backbone.")
    parser.add_argument("--training_epochs", default=10, type=int, help="Number of epochs to train the skipping model")

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    logging.info('Experiment dir : {}'.format(args.output_path))

    fh = logging.FileHandler(os.path.join(args.output_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    device = args.device
    use_cuda=False
    if torch.cuda.is_available() and device != 'cpu':
        device = 'cuda:{}'.format(device)
        logging.info("Running on GPU")
        use_cuda=True
    else:
        logging.info("No device found")
        logging.warning("Device not found or CUDA not available.")
    
    device = torch.device(device)
    
    # Clear GPU cache before starting training
    if use_cuda:
        logging.info("Clearing GPU cache before training...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check available GPU memory
        gpu_id = int(str(device).split(':')[1]) if ':' in str(device) else 0
        free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3  # Free memory in GB
        total_mem = torch.cuda.mem_get_info(gpu_id)[1] / 1024**3  # Total memory in GB
        logging.info(f"GPU {gpu_id} Memory: {free_mem:.2f}GB free out of {total_mem:.2f}GB total")
        
        if free_mem < 2.0:  # Less than 2GB free
            logging.warning(f"Low GPU memory available ({free_mem:.2f}GB). Training may fail.")
    
    initialize_seed(42, use_cuda)

    if args.model not in ['skippingmobilenetv3', 'skippingmobilenetv3_extended']:
        raise ValueError(f"Only skippingmobilenetv3 and skippingmobilenetv3_extended are supported, got {args.model}")
    
    n_subnet = args.output_path.rsplit("_", 1)[1]
    save_path = os.path.join(args.output_path, 'net_{}.stats'.format(n_subnet))

    supernet_path = args.supernet_path
    if args.model_path is not None:
        model_path = args.model_path
    logging.info("Model: %s", args.model)
    
    backbone, res = get_network_search(model=args.model,
                                subnet_path=args.model_path, 
                                supernet=args.supernet_path, 
                                n_classes=args.n_classes, 
                                pretrained=args.pretrained,
                                func_constr=args.func_constr)

    if res is None:
        res = args.resolution

    logging.info(f"DATASET: {args.dataset}")
    logging.info("Resolution: %s", res)
    logging.info("Number of classes: %s", args.n_classes)
    logging.info("Training epochs: %s", args.training_epochs)

    train_loader, val_loader, test_loader = get_data_loaders(dataset=args.dataset, batch_size=args.batch_size, threads=args.n_workers, 
                                            val_split=args.val_split, img_size=res, augmentation=True, eval_test=args.eval_test)
    
    if val_loader is not None:
        n_samples=len(val_loader.dataset)
    else:
        val_loader = test_loader
        n_samples=len(test_loader.dataset)

    print("Train samples: ", len(train_loader.dataset))
    print("Val samples: ", len(val_loader.dataset))

    train_log = Log(log_each=10)
    optimizer = get_optimizer(backbone.parameters(), args.optim, args.learning_rate, args.momentum, args.weight_decay)
    criterion = get_loss('ce')
    scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=args.backbone_epochs)

    if (os.path.exists(os.path.join(args.output_path,'backbone.pth'))):

        backbone, optimizer = load_checkpoint(backbone, optimizer, device, os.path.join(args.output_path,'backbone.pth'))
        logging.info("Loaded checkpoint")
        top1 = validate(val_loader, backbone, device, print_freq=100)/100 #correct?

    else:

        if args.backbone_epochs > 0:
            logging.info("Start training...")
            top1, backbone, optimizer = train(train_loader, val_loader, args.backbone_epochs, backbone, device, optimizer, criterion, scheduler, train_log, ckpt_path=os.path.join(args.output_path,'backbone.pth'))
            logging.info("Training finished")
    
    if args.backbone_epochs == 0:
        top1 = validate(val_loader, backbone, device, print_freq=100)
    logging.info(f"VAL ACCURACY BACKBONE: {np.round(top1*100,2)}")
    if args.eval_test:
        top1_test = validate(test_loader, backbone, device, print_freq=100)
        logging.info(f"TEST ACCURACY BACKBONE: {top1_test}")
    
    results={}
    results['backbone_top1'] = float(np.round(100-top1,2))

    # Load subnet configuration to check for extended gate parameters
    if args.model_path and os.path.exists(args.model_path):
        subnet_config = json.load(open(args.model_path))
        
        # Extract gate parameters if they exist (for extended version)
        gate_hidden_size = subnet_config.get('gate_hidden_size', 32)  # Default to 32
        target_sparsities = subnet_config.get('target_sparsities', [0.5] * 16)  # Default to list of 0.5s
        
        # Filter out zeros and count how many gates will be created
        num_gates_to_create = sum(1 for ts in target_sparsities if ts != 0)
        
        logging.info(f"Using gate parameters: hidden_size={gate_hidden_size}")
        logging.info(f"Target sparsities array (first 10): {target_sparsities[:10]}")
        logging.info(f"Number of gates to create (non-zero targets): {num_gates_to_create}")
    else:
        raise ValueError("Model path not provided or does not exist for skipping model.")
    
    backbone = get_skipping_mobilenetv3(subnet=backbone, subnet_path=args.model_path, res=res, n_classes=args.n_classes,
        gate_type='stable',  # Use specified gate type
        enable_gates=True,
        gate_hidden_size=gate_hidden_size,
        target_sparsities=target_sparsities)  # Pass array

    from ofa.utils.pytorch_utils import count_parameters
    b_params = [count_parameters(backbone)]
    
    # Calculate baseline MACs
    backbone.eval()
    backbone.to(device)
    if hasattr(backbone, 'gates') and len(backbone.gates) > 0:
        dummy_decisions = torch.ones(10, len(backbone.gates), device=device)
        _, _, baseline_macs, _ = backbone.calculate_macs_accurate(
            dummy_decisions, input_size=(3, res, res), device=device
        )
        b_macs = [baseline_macs / 1e6]
        logging.info(f"Baseline MACs: {baseline_macs/1e6:.2f}M")
    else:
        b_macs = [0]  # Fallback estimate
        logging.info("Error: No gates found in skipping model for MAC calculation.")
    
    results['backbone_params'] = b_params
    results['backbone_macs'] = b_macs

    print("Backbone MACS: ", b_macs)
    print("Backbone params: ", b_params)

    # Check if model is already trained
    if os.path.exists(os.path.join(args.output_path, 'bb.pt')):
        logging.info('Model loaded')
        backbone.to(device)
        backbone.load_state_dict(torch.load(
            os.path.join(args.output_path, 'bb.pt'), map_location=device))
        backbone_dict = backbone.state_dict()
    else:
        # Train the skipping model
        logging.info("Start training of the Skipping MobileNetV3...")
        backbone.to(device)
        
        # Separate gate and backbone parameters
        gate_params = []
        backbone_params = []
        for name, param in backbone.named_parameters():
            if 'gates' in name or 'gate_net' in name:
                gate_params.append(param)
            else:
                backbone_params.append(param)
        epochs = args.training_epochs if args.training_epochs > 0 else 1
        # Use SGD optimizer for skipping models
        optimizer = get_optimizer(backbone.parameters(), 'SGD', args.learning_rate, args.momentum, args.weight_decay)
        criterion = get_loss('ce')
        scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=epochs, lr_min=0)
        
        # Training loop
        from tqdm import tqdm
        
        best_accuracy = 0.0
        
        # Log training configuration
        logging.info(f"Starting SkippingMobileNetV3 training for {epochs} epochs...")
        logging.info("Gate strategy: Soft gates 50% of time (no warmup phase)")
        logging.info(f"Number of gates: {len(backbone.gates)}")
        if len(backbone.gates) > 0:
            logging.info(f"Per-gate target sparsities: {backbone.target_sparsities.tolist()}")
        
        for epoch in range(epochs):
                # Training phase
                backbone.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                sparsity_losses = 0.0
                
                # Determine training strategy 
                warmup_phase = False
                use_soft_gates = epoch % 2 == 0  # Use soft gates 50% of time
                
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                
                for batch_idx, (data, target) in enumerate(progress_bar):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output, aux = backbone(data, hard=not use_soft_gates)
                    
                    # Task loss
                    task_loss = criterion(output, target)
                    
                    # Sparsity loss (per-gate with individual targets)
                    sparsity_loss = 0.0
                    if aux and "gate_values" in aux and aux["gate_values"] is not None:
                        # Calculate sparsity weight
                        ramp_progress = min(1.0, epoch / 10.0)
                        current_sparsity_weight = 0.1 + 0.9 * ramp_progress
                        
                        gate_values = aux["gate_values"]  # Shape: (batch, num_gates)
                        
                        # Ensure target_sparsities is on the same device
                        if not hasattr(backbone, '_target_sparsities_device_moved'):
                            backbone.target_sparsities = backbone.target_sparsities.to(device)
                            backbone._target_sparsities_device_moved = True
                        
                        # Per-gate sparsity loss using individual targets
                        for gate_idx in range(gate_values.shape[1]):
                            gate_sparsity = torch.mean(gate_values[:, gate_idx])
                            target_sparsity = backbone.target_sparsities[gate_idx]  # Per-gate target
                            sparsity_diff = gate_sparsity - target_sparsity
                            sparsity_loss += current_sparsity_weight * (sparsity_diff ** 2)
                        
                        # Add diversity loss to prevent gate collapse
                        gate_variance = gate_values.var(dim=1).mean()  # Variance across gates per sample
                        diversity_loss = max(0, 0.01 - gate_variance)
                        sparsity_loss += current_sparsity_weight * diversity_loss
                    
                    total_loss = task_loss + sparsity_loss
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
                    optimizer.step()
                    
                    # Track statistics
                    train_loss += total_loss.item()
                    sparsity_losses += (sparsity_loss if isinstance(sparsity_loss, (int, float)) else sparsity_loss.item())
                    
                    _, predicted = output.max(1)
                    # Ensure target is at least 1D and get batch size
                    if target.dim() == 0:
                        target = target.unsqueeze(0)
                    batch_size = target.size(0)
                    train_correct += predicted.eq(target).sum().item()
                    train_total += batch_size
                
                scheduler.step()
                
                # Calculate epoch statistics
                epoch_loss = train_loss / len(train_loader)
                epoch_acc = 100. * train_correct / train_total
                epoch_sparsity_loss = sparsity_losses / len(train_loader)
                
                # Validation
                backbone.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output, aux = backbone(data, hard=True)
                        
                        # Ensure target is at least 1D and get batch size
                        if target.dim() == 0:
                            target = target.unsqueeze(0)
                        batch_size = target.size(0)
                        val_loss += criterion(output, target).item() * batch_size
                        _, predicted = output.max(1)
                        val_total += batch_size
                        val_correct += predicted.eq(target).sum().item()
                
                # Calculate validation statistics
                val_loss /= val_total
                val_acc = 100. * val_correct / val_total
                
                # Calculate efficiency metrics with accurate MAC counting (like dyn_nas_untrained.py)
                all_gate_decisions = []
                # Collect gate decisions from validation
                backbone.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader):
                        if batch_idx >= 3:  # Sample first 3 batches for efficiency calculation (reduced from 10)
                            break
                        data = data.to(device, non_blocking=True)
                        _, aux_val = backbone(data, hard=True)
                        if aux_val and "gate_decisions" in aux_val and aux_val["gate_decisions"] is not None:
                            all_gate_decisions.append(aux_val["gate_decisions"])
                
                if all_gate_decisions:
                    all_decisions = torch.cat(all_gate_decisions, dim=0)  # (total_samples, num_gates)
                    mac_ratio, sparsity_rate, baseline_macs, real_gated_macs = backbone.calculate_macs_accurate(
                        all_decisions, input_size=(3, res, res), device=device
                    )
                    efficiency_percent = (1 - mac_ratio) * 100
                    avg_skip_prob = sparsity_rate  # Use actual sparsity rate from MAC calculation
                else:
                    sparsity_rate = 0.0
                    efficiency_percent = 0.0
                    baseline_macs = 0
                    real_gated_macs = 0
                    avg_skip_prob = 0.0
                
                # Show gate mode and detailed metrics like dyn_nas_untrained.py
                gate_mode = 'Soft' if use_soft_gates else 'Hard'
                logging.info(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Sparsity Loss: {epoch_sparsity_loss:.4f}, Gate Mode: {gate_mode}')
                if hasattr(backbone, 'gates') and len(backbone.gates) > 0:
                    logging.info(f'  Gates: Sparsity={sparsity_rate:.2%}, Est. MAC Savings={efficiency_percent:.1f}%')
                    
                    # Warning for potential issues (like dyn_nas_untrained.py)
                    if sparsity_rate > 0.8:
                        logging.info(f"  ⚠️  HIGH SPARSITY WARNING: {sparsity_rate:.1%} - Gates may be collapsing!")
                    if val_acc < epoch_acc - 20:
                        logging.info(f"  ⚠️  OVERFITTING WARNING: Train-Val gap = {epoch_acc - val_acc:.1f}%")
                
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
        
        # Training completed - show summary
        logging.info(f"SkippingMobileNetV3 training completed!")
        logging.info(f"Best validation accuracy: {best_accuracy:.2f}%")
        
        # Set state dict for saving
        backbone_dict = backbone.state_dict()
    
    backbone.load_state_dict(backbone_dict)

    if args.save:
        torch.save(backbone.state_dict(), os.path.join(args.output_path, 'bb.pt'))

    # Evaluation
    logging.info("Evaluating skipping model...")
    
    # Accuracy evaluation and gate decision collection
    backbone.eval()
    correct = 0
    total = 0
    all_gate_decisions = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, aux = backbone(data, hard=True)  # Use hard gates for evaluation
            _, predicted = output.max(1)
            # Ensure target is at least 1D and get batch size
            if target.dim() == 0:
                target = target.unsqueeze(0)
            batch_size = target.size(0)
            total += batch_size
            correct += predicted.eq(target).sum().item()
            
            # Collect gate decisions for MAC calculation
            if aux and "gate_decisions" in aux and aux["gate_decisions"] is not None:
                all_gate_decisions.append(aux["gate_decisions"])
    
    accuracy = correct / total
    best_scores = {'global': accuracy}
    
    # Calculate accurate MACs using gate decisions
    if all_gate_decisions:
        all_decisions = torch.cat(all_gate_decisions, dim=0)
        mac_ratio, sparsity_rate, baseline_macs, real_gated_macs = backbone.calculate_macs_accurate(
            all_decisions, input_size=(3, res, res), device=device
        )
        avg_macs = real_gated_macs / 1e6  # Convert to MMAC
        efficiency_percent = (1 - mac_ratio) * 100
        logging.info(f"Skipping model evaluation - Sparsity: {sparsity_rate:.2%}, MAC Savings: {efficiency_percent:.1f}%")
        logging.info(f"Baseline MACs: {baseline_macs/1e6:.2f}M, Gated MACs: {real_gated_macs/1e6:.2f}M")
    else:
        raise ValueError("No gate decisions collected for MAC calculation during evaluation.")
        
    # weights = [1.0]  # Single exit

    # Populate results
    results['avg_macs'] = avg_macs  # Actual MACs with adaptive gating/skipping
    # results['cumulative_threshold'] = False

    results['top1'] = (1-best_scores['global']) * 100 #top1 error
    # results['branch_scores'] = best_scores
    results['params'] = (b_params[-1] if b_params else 0)
    results['macs'] = (b_macs[-1] if b_macs else 0)  # Baseline MACs (maximum possible)
    
    with open(save_path, 'w') as handle:
        json.dump(results, handle)

    logging.info("Results saved to %s", save_path)
    
    # Clean up GPU memory before exiting
    logging.info("Cleaning up GPU memory...")
    del backbone
    del train_loader
    del val_loader
    if use_cuda:
        torch.cuda.empty_cache()
    gc.collect()
    logging.info("GPU memory cleanup completed")
