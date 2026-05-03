import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18

def get_resnet_backbone():
    # 1. Load ResNet-18 from scratch (No pretrained weights)
    base_model = resnet18(weights=None)
    # print("Original ResNet-18 architecture:")
    # print(base_model)

    # 2. Modify the first convolution layer
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Target:   Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 3. Change max-pooling to an Identity layer
    # This prevents the initial 32x32 image from becoming 16x16 immediately
    base_model.maxpool = nn.Identity()

    # 4. Remove the final Fully Connected (FC) layer
    # We want the output after Global Average Pooling (512 dimensions)
    # We replace the FC layer with Identity so it just passes the features through
    base_model.fc = nn.Identity()
    
    # print("Modified ResNet-18 architecture:")
    # print(base_model)
    
    return base_model

class ProjectorHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class SimCLR(nn.Module):
    def __init__(self, config: dict, chkpt_file_pth: str, device):
        super(SimCLR, self).__init__()
        
        self.chkpt_file_pth = chkpt_file_pth
        self.device = device
        
        # Hyperparameters from config
        lr = float(config.get("learning_rate", 3e-4))
        weight_decay = float(config.get("weight_decay", 1e-6))
        self.temperature = float(config.get("temperature", 0.5))
        
        # ==========================================
        # 1. BASE ENCODER (Modified ResNet-18)
        # ==========================================
        self.backbone = resnet18(weights=None)
        
        # Modify conv1: 3x3 kernel, stride=1, padding=1 (PDF Instruction)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Modify maxpool: Identity (Avoid downsampling)
        self.backbone.maxpool = nn.Identity()
        # Remove fc layer: Identity (We want the 512-dim avg pool output)
        self.backbone.fc = nn.Identity()
        
        # ==========================================
        # 2. PROJECTOR HEAD (2-Layer MLP)
        # ==========================================
        # PDF Instruction: 512 -> 512 hidden nodes -> 128 outputs
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.projector = nn.Identity() if config.get("no_projector", False) else self.projector
        
        # Optimizer integrated RL-style
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)

    def forward(self, x):
        """
        Returns both the representation 'h' (for evaluation) 
        and the projection 'z' (for contrastive loss).
        """
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

    def learn(self, x_i, x_j):
        """
        The NT-Xent loss logic, exactly mirroring the TensorFlow objective.py code from the original SimCLR paper.
        Reference: https://github.com/google-research/simclr
        """
        # Move inputs to device
        x_i = x_i.to(self.device)
        x_j = x_j.to(self.device)
        
        # Get projections
        _, z_i = self.forward(x_i)  # Hidden1 in TF code
        _, z_j = self.forward(x_j)  # Hidden2 in TF code
        
        # 1. Normalize hidden vectors
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        batch_size = z_i.shape[0]
        LARGE_NUM = 1e9
        
        # Mask to penalize (z_i, z_i) and (z_j, z_j) comparisons
        masks = torch.eye(batch_size, device=self.device)
        
        # ==========================================
        # 2. The 4-MatMul logic (from objective.py)
        # ==========================================
        # logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
        logits_aa = torch.matmul(z_i, z_i.T) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        
        # logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
        logits_bb = torch.matmul(z_j, z_j.T) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        
        # logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
        logits_ab = torch.matmul(z_i, z_j.T) / self.temperature
        
        # logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
        logits_ba = torch.matmul(z_j, z_i.T) / self.temperature
        
        # ==========================================
        # 3. Cross Entropy Loss
        # ==========================================
        # In the TF code: tf.concat([logits_ab, logits_aa], 1)
        # The positive pair is at the exact same index `k` in the first block (logits_ab).
        # Therefore, the target class index for row `k` is just `k`.
        labels = torch.arange(batch_size, device=self.device)
        
        # loss_a = tf.losses.softmax_cross_entropy(...)
        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        
        # loss_b = tf.losses.softmax_cross_entropy(...)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        
        # Total loss
        loss = loss_a + loss_b

        # ==========================================
        # 4. Optimizer Step (RL Style)
        # ==========================================
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: nn.utils.clip_grad_norm_(self.parameters(), max_norm=...)
        self.optimizer.step()
        
        return loss.item()

    def save_model(self):
        print("...saving checkpoint...")
        torch.save({"model": self.state_dict(), 
                    "optimizer": self.optimizer.state_dict()}, 
                   self.chkpt_file_pth)

    def load_model(self):
        print("...loading checkpoint...")
        checkpoint = torch.load(self.chkpt_file_pth, map_location=self.device)
        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class ResNetClassifier(nn.Module):
    def __init__(self, config: dict, simclr_chkpt_path: str, device):
        super(ResNetClassifier, self).__init__()
        self.device = device
        
        # Hyperparameters (PDF says lr=1e-3 for probing)
        lr = float(config.get("learning_rate", 1e-3))
        weight_decay = float(config.get("weight_decay", 1e-6))
        
        # Modes: "linear_probe", "supervised", "random_init", OR "linear_probe_projector"
        self.mode = config.get("mode", "linear_probe") 
        
        # ==========================================
        # 1. BASE ENCODER
        # ==========================================
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # ==========================================
        # 2. PROJECTOR HEAD (Only used for the ablation)
        # ==========================================
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        # ==========================================
        # 3. CLASSIFICATION HEAD
        # ==========================================
        # If we use the projector representation, the input dimension is 128. Otherwise, 512.
        ###### Comment and uncomment if using CIFAR-100 instead of CIFAR-10 ######
        if self.mode == "linear_probe_projector":
            self.head = nn.Linear(128, 10) # 128 -> CIFAR-10
            # self.head = nn.Linear(128, 100) # 128 -> CIFAR-100
        else:
            self.head = nn.Linear(512, 10) # 512 -> CIFAR-10
            # self.head = nn.Linear(128, 100) # 128 -> CIFAR-100

        # ==========================================
        # 4. MODE LOGIC (Load & Freeze)
        # ==========================================
        if self.mode == "linear_probe":
            print(f"--> [Linear Probe] Loading SSL weights and freezing backbone...")
            checkpoint = torch.load(simclr_chkpt_path, map_location=self.device)
            backbone_weights = {k.replace('backbone.', ''): v 
                                for k, v in checkpoint["model"].items() if k.startswith('backbone.')}
            self.backbone.load_state_dict(backbone_weights)
            
            # Freeze the backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        elif self.mode == "linear_probe_projector":
            print(f"--> [Projector Ablation] Loading SSL backbone + projector and freezing both...")
            checkpoint = torch.load(simclr_chkpt_path, map_location=self.device)
            
            # Load backbone
            backbone_weights = {k.replace('backbone.', ''): v 
                                for k, v in checkpoint["model"].items() if k.startswith('backbone.')}
            self.backbone.load_state_dict(backbone_weights)
            
            # Load projector
            projector_weights = {k.replace('projector.', ''): v 
                                 for k, v in checkpoint["model"].items() if k.startswith('projector.')}
            self.projector.load_state_dict(projector_weights)
            
            # Freeze both backbone AND projector
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = False

        elif self.mode == "random_init":
            print(f"--> [Random Init] Freezing random backbone...")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        elif self.mode == "supervised":
            print(f"-->[Supervised] Training entire network from scratch...")
            pass 

        self.to(self.device)

        # ==========================================
        # 5. OPTIMIZER
        # ==========================================
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    def forward(self, x):
        h = self.backbone(x)
        
        # If testing the projector ablation, pass 'h' through the projector to get 'z' (128 dims)
        if self.mode == "linear_probe_projector":
            z = self.projector(h)
            logits = self.head(z)
        else:
            # Standard probing/supervised uses 'h' (512 dims) directly
            logits = self.head(h)
            
        return logits

    def learn(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        predictions = torch.argmax(logits, dim=1)
        acc = (predictions == y).float().mean().item()
        
        return loss.item(), acc

if __name__ == "__main__":
    get_resnet_backbone()
