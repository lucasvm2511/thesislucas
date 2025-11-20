import torch
from torch import nn
import os
import sys
from torchprofile import profile_macs

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
# from base import BranchModel
from ofa.imagenet_classification.networks import MobileNetV3
from ofa.utils import MyGlobalAvgPool2d
def get_block_channels(blk):
    """Extract input/output channels and stride from a block"""
    convs = [m for m in blk.modules() if isinstance(m, nn.Conv2d)]
    in_ch = convs[0].in_channels
    out_ch = convs[-1].out_channels
    stride = convs[0].stride[0]
    return in_ch, out_ch, stride


class StableGate(nn.Module):
    """Stable sigmoid-based gate with proper initialization and straight-through estimation."""
    
    def __init__(self, in_ch, hidden=32, temperature=1.0):
        super().__init__()
        # Smaller network to reduce overhead
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_ch, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden, 1)
        
        self.temperature = temperature
        self.register_buffer('training_step', torch.tensor(0))
        
        # Proper initialization for stable training
        self._init_weights()
    
    def _init_weights(self):
        # Use smaller initialization to prevent saturation
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0.0)
        # Initialize final layer bias moderately negative for balanced sparsity
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, -0.7)  # Start moderately closed, allow learning
    
    def forward(self, x, hard=True, return_logit=False):
        temperature = self.temperature
        
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        
        # Use BatchNorm if batch_size > 1, else skip it
        if x.size(0) > 1:
            x = self.bn(x)
        else:
            # For batch_size=1, just apply instance normalization manually
            if self.training:
                # During training with batch_size=1, use learned BN params if available
                if self.bn.running_mean is not None:
                    x = (x - self.bn.running_mean) / torch.sqrt(self.bn.running_var + self.bn.eps)
                    x = x * self.bn.weight + self.bn.bias
        
        x = self.relu(x)
        x = self.dropout(x)
        logit = self.fc2(x)
        
        # Apply temperature scaling
        scaled_logit = logit / temperature 
        
        # Get probabilities
        prob = torch.sigmoid(scaled_logit)
        
        # Hard decision
        hard_decision = (prob > 0.5).float()
        
        if hard:
            # Straight-through estimator: forward uses hard, backward uses soft
            gate_value = hard_decision + prob - prob.detach()
        else:
            gate_value = prob
        
        # Update training step counter
        if self.training:
            self.training_step += 1
        
        if return_logit:
            return gate_value, prob, hard_decision, logit
        return gate_value, prob, hard_decision


class FinalClassifier(nn.Module):
          
        def __init__(self, final_expand_layer, feature_mix_layer, classifier):
            super(FinalClassifier, self).__init__()
            self.final_expand_layer = final_expand_layer
            self.feature_mix_layer = feature_mix_layer
            self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)
            self.classifier = classifier
            
        def forward(self, x):
            x = self.final_expand_layer(x)
            x = self.global_avg_pool(x)  # global average pooling
            x = self.feature_mix_layer(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x 


class SkippingMobileNetV3(nn.Module):
    """
    MobileNetV3 with input-dependent block skipping using Gumbel Softmax.
    
    This model uses Gumbel Softmax to make differentiable skip decisions based on
    the input features, allowing the model to learn which blocks to skip
    in a truly input-dependent manner.
    """

    def __init__(self, first_conv, blocks, depth, gate_type="stable", temperature=1.0, n_classes=None, enable_gates=True, gate_hidden_sizes=None, target_sparsities=None):
        """
        Initialize Skipping MobileNetV3 model with per-block target sparsities controlling gate placement.
        
        Args:
            first_conv: Initial convolution layer
            blocks: List of neural network blocks (e.g., 20 inverted residual blocks)
            depth: Number of blocks per group [3,4,2,5] - defines grouping structure
            gate_type: Type of gate to use ("stable" or "conv")
            temperature: Temperature parameter for gates
            n_classes: Number of classes for classification (if None, no classifier is added)
            enable_gates: Whether to enable gates
            gate_hidden_sizes: Array of hidden layer sizes for gates (one per potential gate position)
            target_sparsities: Array of target sparsities per block (0 = no gate, 0.3/0.5/0.7 = sparsity target)
                              Length should be <= number of eligible blocks
        """
        super(SkippingMobileNetV3, self).__init__()
        
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.depth = depth
        self.n_blocks = len(blocks)
        self.gate_type = gate_type
        self.temperature = temperature
        self.enable_gates = enable_gates
        
        # Gate parameters
        self.gate_hidden_sizes_config = gate_hidden_sizes if gate_hidden_sizes is not None else []
        self.target_sparsities_config = target_sparsities if target_sparsities is not None else []
        
        # Try to estimate the number of output channels from the last block
        if hasattr(blocks[-1], 'out_channels'):
            out_channels = blocks[-1].out_channels
        elif hasattr(blocks[-1], 'conv') and hasattr(blocks[-1].conv, 'out_channels'):
            out_channels = blocks[-1].conv.out_channels
        # else:
        #     # Default assumption
        #     out_channels = 128
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, n_classes)
        )
        
        # Create gates
        self.gates = nn.ModuleList()
        self.gate_indices = []  # Track which blocks have gates
        self.target_sparsities = []  # Store per-gate target sparsities (non-zero values only)
        
        if self.enable_gates:
            # Determine which blocks are eligible for gates based on architecture
            eligible_blocks = []
            for i, blk in enumerate(self.blocks):
                try:
                    in_ch, out_ch, stride = get_block_channels(blk)
                    # Only add gates to residual blocks (stride=1, same channels)
                    if stride == 1 and in_ch == out_ch:
                        eligible_blocks.append((i, in_ch))
                except Exception as e:
                    print(f"Warning: Could not analyze block {i}: {e}")
                    continue
            
            # Use target_sparsities array to determine which blocks get gates
            # 0 = no gate, non-zero = create gate with that sparsity target
            for idx, (block_idx, in_ch) in enumerate(eligible_blocks):
                # If we have more eligible blocks than sparsity values, ignore extra blocks
                if idx >= len(self.target_sparsities_config):
                    break
                
                target_sparsity = self.target_sparsities_config[idx]
                
                # Skip if target_sparsity is 0 (no gate for this block)
                if target_sparsity == 0:
                    continue
                
                # Get hidden size for this gate (use default if not provided)
                gate_hidden_size = self.gate_hidden_sizes_config[idx] if idx < len(self.gate_hidden_sizes_config) else 32
                
                try:
                    # Create gate for this block with per-gate hidden size
                    if self.gate_type == "stable":
                        gate = StableGate(in_ch, hidden=gate_hidden_size, temperature=temperature)

                    self.gates.append(gate)
                    self.gate_indices.append(block_idx)
                    self.target_sparsities.append(target_sparsity)  # Store non-zero target
                except Exception as e:
                    print(f"Warning: Could not create gate for block {block_idx}: {e}")
                    continue
            
            if len(self.target_sparsities) > 0:
                self.target_sparsities = torch.tensor(self.target_sparsities, dtype=torch.float32)

        print("Created input-dependent skipping MobileNetV3")
    
    def forward(self, x, hard=True, return_logits=False, true_skip_inference=False):
        """
        Forward pass with stable gate application, similar to MobileNetV3DynamicGated.
        Returns logits and auxiliary information for monitoring and loss calculation.
        """
        # Initialize tracking variables
        gate_values = []  # Continuous gate values [0,1] for loss computation
        gate_probs = []   # Raw probabilities before hard decision
        gate_decisions = []  # Hard binary decisions for actual computation
        gate_logits = [] if return_logits else None
        
        # Process input through first conv
        x = self.first_conv(x)
        
        # Process each block
        gate_idx = 0
        for block_idx, block in enumerate(self.blocks):
            if block_idx in self.gate_indices and self.enable_gates:
                # This block has a gate
                gate = self.gates[gate_idx]
                gate_idx += 1
                
                # Get gate decision
                if return_logits:
                    gate_val, gate_prob, gate_hard, gate_logit = gate(
                        x, hard=hard, return_logit=True
                    )
                    gate_logits.append(gate_logit)
                else:
                    gate_val, gate_prob, gate_hard = gate(
                        x, hard=hard
                    )
                
                # Store gate information as (B,1) to preserve batch dim even when B=1
                gate_values.append(gate_val.view(gate_val.size(0), 1))  # (B,1)
                gate_probs.append(gate_prob.view(gate_prob.size(0), 1))  # (B,1)
                gate_decisions.append(gate_hard.view(gate_hard.size(0), 1))  # (B,1)
                
                # TRUE SKIPPING for batch_size=1 (actual speedup)
                if hard and x.size(0) == 1:
                    if gate_hard.item() > 0.5:  # Gate open - execute block
                        block_output = block(x)
                        if block_output.shape == x.shape:
                            x = block_output + x  # Residual connection
                        else:
                            x = block_output
                    else:
                        # Gate closed - SKIP BLOCK ENTIRELY (true speedud)
                        pass  # x unchanged
                else:
                    # BATCHED MODE: Gated residual (no actual speedup, but works for any batch size)
                    block_output = block(x)
                    
                    if block_output.shape == x.shape:
                        # Residual connection: output = gate * block(x) + (1-gate) * x
                        gate_reshaped = gate_val.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
                        x = gate_reshaped * block_output + (1 - gate_reshaped) * x
                    else:
                        # Non-residual block: always use output (shape-changing blocks)
                        x = block_output
            else:
                # No gate - always execute block
                x = block(x)
        
        x = self.classifier(x)
        
        # Compute statistics
        if gate_values:
            # Concatenate gate outputs along the gate dimension -> (B, num_gates)
            gate_values_tensor = torch.cat(gate_values, dim=1)      # (B, num_gates)
            gate_probs_tensor = torch.cat(gate_probs, dim=1)        # (B, num_gates)
            gate_decisions_tensor = torch.cat(gate_decisions, dim=1)  # (B, num_gates)
            
            # Compute statistics
            mean_gate_prob = gate_probs_tensor.mean().item()
            mean_gate_value = gate_values_tensor.mean().item() 
            sparsity_rate = (1 - gate_decisions_tensor.float()).mean().item()  # Fraction closed
            
            aux = {
                "gate_values": gate_values_tensor,      # For loss computation
                "gate_probs": gate_probs_tensor,        # For monitoring  
                "gate_decisions": gate_decisions_tensor.detach(), # For MAC computation (detached)
                "mean_gate_prob": mean_gate_prob,
                "mean_gate_value": mean_gate_value,
                "sparsity_rate": sparsity_rate,
                "num_gates": len(gate_values)
            }
            
            if gate_logits is not None:
                aux["gate_logits"] = gate_logits
        else:
            # No gates in model
            aux = {
                "gate_values": None,
                "gate_probs": None, 
                "gate_decisions": None,
                "mean_gate_prob": 0.0,
                "mean_gate_value": 0.0,
                "sparsity_rate": 0.0,
                "num_gates": 0
            }
        
        return x, aux
    
    # def forward_batch_true_skip(self, x, hard=True):
    #     """
    #     Batch inference with true skipping by processing samples individually.
    #     Less efficient for batches but provides true computational savings.
    #     """
    #     batch_size = x.size(0)
    #     all_logits = []
    #     all_aux = []
        
    #     # Process each sample individually for true skipping
    #     for i in range(batch_size):
    #         sample = x[i:i+1]  # Keep batch dimension
    #         logits_i, aux_i = self.forward(sample, hard=hard, true_skip_inference=True)
    #         all_logits.append(logits_i)
    #         all_aux.append(aux_i)
        
    #     # Concatenate results
    #     logits = torch.cat(all_logits, dim=0)
        
    #     # Aggregate auxiliary information
    #     if all_aux[0]["gate_values"] is not None:
    #         gate_values = torch.cat([aux["gate_values"] for aux in all_aux], dim=0)
    #         gate_probs = torch.cat([aux["gate_probs"] for aux in all_aux], dim=0)
    #         gate_decisions = torch.cat([aux["gate_decisions"] for aux in all_aux], dim=0)
            
    #         aux = {
    #             "gate_values": gate_values,
    #             "gate_probs": gate_probs,
    #             "gate_decisions": gate_decisions,
    #             "mean_gate_prob": gate_probs.mean().item(),
    #             "mean_gate_value": gate_values.mean().item(),
    #             "sparsity_rate": (1 - gate_decisions.float()).mean().item(),
    #             "num_gates": all_aux[0]["num_gates"]
    #         }
    #     else:
    #         aux = all_aux[0]  # No gates case
            
    #     return logits, aux
    
    def calculate_macs_accurate(self, gate_decisions, input_size, device='cuda'):
        """
        Real MAC calculation using torchprofile for consistency with get_net_info.
        Uses the same profiling library to ensure exact MAC count matching.
        """
        if gate_decisions is None or len(self.gates) == 0:
            return 1.0, 0.0, 0, 0  # No gates, full MAC usage
        
        # gate_decisions shape: (total_samples, num_gates)
        # Example: (96, 10) if 3 batches of 32 samples
        
        # Average across ALL samples to get per-gate statistics
        avg_gate_decisions = gate_decisions.float().mean(dim=0)  # Shape: (num_gates,)
        # Result: [0.85, 0.42, 0.73, 0.91, 0.28, 0.65, 0.53, 0.88, 0.47, 0.71]
        #         ↑ Gate 0 was open 85% of the time across all 96 samples
        #         ↑ Gate 1 was open 42% of the time across all 96 samples
        #         etc.

        # Sparsity rate = fraction of gates that are closed (0)
        sparsity_rate = (1 - gate_decisions.float()).mean().item()
        
        # Use same input format as get_net_info: batch_size=1
        dummy_input = torch.randn(1, *input_size).to(device)
        
        self.eval()
        with torch.no_grad():
            # 1. Baseline MACs (No gates) - matches get_net_info
            self.enable_gates = False # Disable gates for baseline
            baseline_macs = profile_macs(self, dummy_input)
            
            # 2. Measure MAC cost of each individual gated block using torchprofile
            block_macs = {}
            gate_macs = {}
            
            # Measure each gated block individually
            gate_idx = 0
            for block_idx in self.gate_indices:
                block = self.blocks[block_idx]
                gate = self.gates[gate_idx]
                gate_idx += 1
                
                # Get input to this block by running partial model
                x = dummy_input
                x = self.first_conv(x)
                for i, blk in enumerate(self.blocks):
                    if i == block_idx:
                        break
                    x = blk(x)
                
                # Measure block MAC cost using torchprofile
                block_macs[block_idx] = profile_macs(block, x)
                
                # Measure gate MAC cost using torchprofile
                gate_macs[block_idx] = profile_macs(gate, x)
            
            
            # Subtract saved MACs from closed gates, add gate overhead
            total_gate_overhead = 0
            total_saved_macs = 0
            
            for i, block_idx in enumerate(self.gate_indices):
                gate_open_prob = avg_gate_decisions[i].item()  # Probability gate is open
                gate_closed_prob = 1.0 - gate_open_prob       # Probability gate is closed
                
                # Gate overhead (always paid)
                gate_overhead = gate_macs[block_idx]
                total_gate_overhead += gate_overhead
                
                # Block MACs saved when gate is closed
                block_saved = block_macs[block_idx] * gate_closed_prob
                total_saved_macs += block_saved
                
            # Real gated MACs = baseline + gate_overhead - saved_macs
            real_gated_macs = baseline_macs + total_gate_overhead - total_saved_macs
            
        # Calculate actual MAC usage ratio
        if baseline_macs > 0:
            mac_usage_ratio = real_gated_macs / baseline_macs
            mac_savings_percent = (1 - mac_usage_ratio) * 100
        else:
            mac_usage_ratio = 1.0
            mac_savings_percent = 0.0
            
        total_blocks = len(self.blocks) 
        gated_blocks = len(self.gates)
        
        # print(f"  MAC Calc: {gated_blocks}/{total_blocks} blocks gated")
        # print(f"  Baseline MACs: {baseline_macs:,}")
        # print(f"  Gate Overhead: +{total_gate_overhead:,.0f} ({total_gate_overhead/baseline_macs*100:.2f}%)")
        # print(f"  Block Savings: -{total_saved_macs:,.0f} ({total_saved_macs/baseline_macs*100:.2f}%)")
        # print(f"  Real Gated MACs: {real_gated_macs:,.0f}")
        # print(f"  Real MAC Savings: {mac_savings_percent:.1f}%")
        # print(f"  Sparsity Rate: {sparsity_rate:.1%}")
        
        self.enable_gates = True  # Ensure gates are re-enabled after MAC calculation
        return mac_usage_ratio, sparsity_rate, baseline_macs, int(real_gated_macs)
    
    
    def _analyze_block_compatibility(self, input_shape=(1, 3, 32, 32)):
        """
        Analyze which blocks can be safely skipped (no dimension changes).
        
        Returns:
            List of booleans indicating which blocks can be skipped
        """
        was_training = self.training
        self.eval()
        skippable_blocks = []
        
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.randn(input_shape, device=device)
            x = self.first_conv(x)
            
            for block in self.blocks:
                shape_before = x.shape
                x = block(x)
                # Block can be skipped if input and output have same shape
                skippable_blocks.append(shape_before == x.shape)
        
        self.train(was_training)  # Restore original mode
        return skippable_blocks
    
    def get_skippable_blocks(self):
        """Get list of blocks that can be safely skipped (cached)."""
        if not hasattr(self, '_skippable_blocks'):
            self._skippable_blocks = self._analyze_block_compatibility()
        return self._skippable_blocks
    
    
    def get_gate_probabilities(self, sample_input=None):
        """
        Get gate probabilities for each gated block.
        
        Args:
            sample_input: Input tensor to compute gate probabilities.
                         If None, returns placeholder strings.
        
        Returns:
            List of gate probabilities or placeholder strings
        """
        if not self.enable_gates or len(self.gates) == 0:
            return [0.0] * self.n_blocks
        
        if sample_input is None:
            return ["input-dependent"] * len(self.gates)
        
        # Compute actual probabilities with sample input
        gate_probs = []
        with torch.no_grad():
            self.eval()
            x = self.first_conv(sample_input)
            gate_idx = 0
            
            for block_idx, block in enumerate(self.blocks):
                if block_idx in self.gate_indices:
                    # Get gate probability
                    gate = self.gates[gate_idx]
                    _, gate_prob, _ = gate(x, hard=False)
                    gate_probs.append(gate_prob.mean().item())
                    gate_idx += 1
                
                # Update features for next block
                x = block(x)
        
        return gate_probs
    
    def n_branches(self):
        return 1






