import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
from torchprofile import profile_macs

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
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
    """Stable sigmoid-based gate with straight-through estimation.
    
    Supports dimension mismatches via learnable projection when in_ch != out_ch or stride != 1.
    """
    
    def __init__(self, in_ch, out_ch=None, stride=1, hidden=32, temperature=1.0):
        super().__init__()
        # If out_ch not specified, assume same as in_ch
        if out_ch is None:
            out_ch = in_ch
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.has_projection = (in_ch != out_ch) or (stride != 1)
        
        # Projection layer to match dimensions when needed
        if self.has_projection:
            self.projection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        
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
        # Initialize convolutional layer
        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize projection layer if present
        if self.has_projection:
            nn.init.kaiming_normal_(self.projection[0].weight, mode='fan_out', nonlinearity='relu')
        
        # Use smaller initialization to prevent saturation
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0.0)
        # Initialize final layer bias moderately negative for balanced sparsity
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, -0.7)  # Start moderately closed, allow learning
    
    def forward(self, x, hard=True, return_logit=False):
        temperature = self.temperature
        
        # Store input for potential projection
        identity = x

        
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
            return gate_value, prob, hard_decision, logit, identity
        return gate_value, prob, hard_decision, identity

class ConvGate(nn.Module):
    def __init__(self, in_ch, out_ch=None, stride=1, hidden=32, temperature=1.0):
        super().__init__()

        if out_ch is None:
            out_ch = in_ch

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.has_projection = (in_ch != out_ch) or (stride != 1)
        self.temperature = temperature

        # --- Progressive spatial downsampling --------------------------------
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu2 = nn.ReLU(inplace=True)

        # --- Projection block -------------------------------------------------
        if self.has_projection:
            self.projection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        # --- Gate head: decision on spatial features -------------------------
        # Final 1x1 conv to produce gate logit (keeps spatial structure)
        self.gate_conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)  # Collapse to scalar at the end

        # counter
        self.register_buffer("training_step", torch.tensor(0))

        self._init_weights()

    # ---------------------------------------------------------------------- #
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        if self.has_projection:
            nn.init.kaiming_normal_(self.projection[0].weight, mode='fan_out', nonlinearity='relu')

        nn.init.normal_(self.gate_conv.weight, 0, 0.01)
        nn.init.constant_(self.gate_conv.bias, -0.7)  # moderately closed start

    # ---------------------------------------------------------------------- #
    def forward(self, x, hard=True, return_logit=False):
        identity = x

        # --- Progressive downsampling: 
        x = self.maxpool(x)      # Reduce by 2x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)         # Reduce by another 2x with stride
        x = self.bn2(x)
        x = self.relu2(x)

        # --- Decision on spatial features ------------------------------------
        logit_map = self.gate_conv(x)  # (B, 1, H', W') - spatial gate map
        logit = self.pool(logit_map)   # (B, 1, 1, 1) - global average
        logit = logit.view(logit.size(0), 1)  # (B, 1)

        # temperature scaling
        prob = torch.sigmoid(logit / self.temperature)

        # straight-through binary gate
        hard_decision = (prob > 0.5).float()
        gate = hard_decision + prob - prob.detach() if hard else prob

        if self.training:
            self.training_step += 1

        if return_logit:
            return gate, prob, hard_decision, logit, identity
        return gate, prob, hard_decision, identity

class AttentionGate(nn.Module):
    """
    Attention-driven gating module combining:
    - Channel Attention (SE)
    - Spatial Attention (CBAM-style)
    - Final scalar gate (sigmoid or STE)
    """

    def __init__(self, in_ch, out_ch=None, stride=1, reduction=16, hidden=32, temperature=1.0):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.has_projection = (in_ch != out_ch) or (stride != 1)
        self.temperature = temperature
        
        # Projection layer to match dimensions when needed
        if self.has_projection:
            self.projection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        # -----------------------
        # Channel Attention (SE)
        # -----------------------
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

        # -----------------------
        # Spatial Attention (CBAM)
        # -----------------------
        # compress across channels: max + avg pooling
        self.spatial_compress = lambda x: torch.cat(
            [torch.max(x, dim=1, keepdim=True)[0],
             torch.mean(x, dim=1, keepdim=True)],
            dim=1
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # -----------------------
        # Gate head: combine attentions → scalar probability
        # -----------------------
        # vector features: channel_att (C), spatial_att (H*W pooled into scalar)
        self.fc = nn.Sequential(
            nn.Linear(in_ch + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

        # bias: start moderately closed
        nn.init.constant_(self.fc[-1].bias, -0.7)

    def forward(self, x, hard=True, return_logit=False):
        B, C, H, W = x.shape
        
        # Store input for potential projection
        identity = x

        # ----------------------------------------
        # CHANNEL ATTENTION (SE)
        # ----------------------------------------
        ch_vec = self.avg_pool(x).view(B, C)           # (B, C)
        ch_att = self.channel_fc(ch_vec)               # (B, C)
        x_ch = x * ch_att.view(B, C, 1, 1)             # channel refined feature

        # ----------------------------------------
        # SPATIAL ATTENTION (CBAM)
        # ----------------------------------------
        sp_in = self.spatial_compress(x_ch)            # (B, 2, H, W)
        sp_att = torch.sigmoid(self.spatial_conv(sp_in))  # (B, 1, H, W)
        x_sp = x_ch * sp_att                           # combined refined feature

        # For gating, compress spatial map to a scalar (global importance)
        sp_scalar = F.adaptive_avg_pool2d(sp_att, 1).view(B, 1)  # (B, 1)

        # ----------------------------------------
        # GATE HEAD
        # ----------------------------------------
        gate_in = torch.cat([ch_vec, sp_scalar], dim=1)  # (B, C+1)
        logit = self.fc(gate_in)                         # (B, 1)
        prob = torch.sigmoid(logit / self.temperature)   # scaled probability

        # Straight-through binary decision
        hard_decision = (prob > 0.5).float()
        gate = hard_decision + prob - prob.detach() if hard else prob

        if return_logit:
            return gate, prob, hard_decision, logit, identity
        return gate, prob, hard_decision, identity

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
            # Skip block 0 (fixed OFA block) and allow all other blocks
            eligible_blocks = []
            for i, blk in enumerate(self.blocks):
                # Skip block 0 - it's the fixed architectural block added by OFA
                if i == 0:
                    continue
                    
                try:
                    in_ch, out_ch, stride = get_block_channels(blk)
                    # All blocks (except block 0) are eligible - gates handle dimension changes
                    eligible_blocks.append((i, in_ch, out_ch, stride))
                except Exception as e:
                    print(f"Warning: Could not analyze block {i}: {e}")
                    continue
            
            print(f"Debug: Total blocks: {len(self.blocks)}, Eligible blocks: {len(eligible_blocks)}")
            print(f"Debug: Target sparsities config length: {len(self.target_sparsities_config)}")
            
            # Use target_sparsities array to determine which blocks get gates
            # 0 = no gate, non-zero = create gate with that sparsity target
            gates_created = 0
            gates_skipped_zero = 0
            gates_failed = 0
            
            for idx, (block_idx, in_ch, out_ch, stride) in enumerate(eligible_blocks):
                # Map block indices to config array positions
                # Block 0 is skipped, so block 1 maps to config[0], block 2 to config[1], etc.
                config_idx = block_idx - 1  # Offset by 1 since block 0 is skipped
                
                if config_idx >= len(self.target_sparsities_config):
                    print(f"Debug: Block {block_idx} has no config (config_idx={config_idx} beyond array length)")
                    continue  # No config for this block
                
                target_sparsity = self.target_sparsities_config[config_idx]
                
                # Skip if target_sparsity is 0 (no gate for this block)
                if target_sparsity == 0:
                    gates_skipped_zero += 1
                    continue
                
                # Get hidden size for this gate (use default if not provided)
                gate_hidden_size = self.gate_hidden_sizes_config[config_idx] if config_idx < len(self.gate_hidden_sizes_config) else 32
                
                try:
                    # Create gate for this block with dimension handling
                    if self.gate_type == "stable":
                        gate = StableGate(in_ch, out_ch=out_ch, stride=stride, 
                                        hidden=gate_hidden_size, temperature=temperature)
                    elif self.gate_type == "conv":
                        gate = ConvGate(in_ch, out_ch=out_ch, stride=stride,
                                       hidden=gate_hidden_size, temperature=temperature)
                    elif self.gate_type == "attention":
                        gate = AttentionGate(in_ch, out_ch=out_ch, stride=stride,
                                           hidden=gate_hidden_size, temperature=temperature)
                    else:
                        raise ValueError(f"Unknown gate_type: {self.gate_type}")

                    self.gates.append(gate)
                    self.gate_indices.append(block_idx)
                    self.target_sparsities.append(target_sparsity)  # Store non-zero target
                    gates_created += 1
                    print(f"Debug: Created {self.gate_type} gate for block {block_idx} (in_ch={in_ch}, out_ch={out_ch}, stride={stride}, target={target_sparsity})")
                except Exception as e:
                    print(f"Warning: Could not create gate for block {block_idx}: {e}")
                    gates_failed += 1
                    continue
            
            print(f"Debug: Gates created: {gates_created}, Skipped (sparsity=0): {gates_skipped_zero}, Failed: {gates_failed}")
            
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
                    gate_val, gate_prob, gate_hard, gate_logit, gate_input = gate(
                        x, hard=hard, return_logit=True
                    )
                    gate_logits.append(gate_logit)
                else:
                    gate_val, gate_prob, gate_hard, gate_input = gate(
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
                        # Gate closed - SKIP BLOCK, use projection if needed
                        if gate.has_projection:
                            x = gate.projection(x)
                        # else: x unchanged (same dimensions)
                else:
                    # BATCHED MODE:  (no actual speedup, but works for any batch size)
                    block_output = block(x)
                    
                    # Check if dimensions match between input and output
                    if block_output.shape == x.shape:
                        gate_reshaped = gate_val.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
                        x = gate_reshaped * block_output + (1 - gate_reshaped) * x
                    else:
                        # Dimensions differ - can't use true gating, always use block output
                        # (Projection would be needed but doesn't provide compute savings in batch mode)
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
    
    
    def n_branches(self):
        return 1