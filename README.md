
# Self-Supervised Learning (SimCLR) on CIFAR-10
### NYCU Undergraduate AI Capstone - Project #2 (Spring 2026)

## 📁 Dataset and Training Graphs
The raw training data (exported from Weights & Biases) is located in `dataset/training_data_csv/`. The generated PNG training graphs comparing different ablations (Temperature, Batch Size, Augmentation, etc.) can be found in the `figures/` directory. 

---

## 🛠️ Project Scope
This project focuses on implementing and evaluating **Self-Supervised Learning (SSL)** using the **SimCLR** framework on the CIFAR-10 dataset.

1. **The Dataset:** CIFAR-10, utilizing powerful data augmentation pipelines (random crops, color jitter, horizontal flips, etc.) to generate twin views for contrastive learning.
2. **The Models:** 
    * **Backbone:** A modified **ResNet-18** (adjusted initial convolution and pooling for 32x32 images).
    * **Projector:** A 2-layer MLP projection head mapping 512-dimensional representations to a 128-dimensional contrastive space.
3. **Training & Ablation:** The core model trains using the normalized temperature-scaled cross-entropy (**NT-Xent**) loss. The setup supports numerous ablation studies including:
    * Temperature scaling comparisons
    * Batch size impact
    * Data augmentation strategies
    * The necessity of the nonlinear projector head
4. **Evaluation:** Representation quality is tracked continuously using a **k-Nearest Neighbor (kNN)** monitor. Final performance is evaluated by training a linear classifier (Linear Probe) on the frozen representations and comparing it against fully supervised baselines and random initializations.

---

## 📂 Repository Structure
* `/dataset`: Automatic download location for CIFAR-10. Also contains `training_data_csv/` with raw logged run data.
* `/figures`: Visualization outputs (Baseline training curves, ablation comparisons, evaluation accuracy).
* `/models`: Saved PyTorch model checkpoints (e.g., `best_simclr_model.pth`).
* `/src`:
    * `training_models.py`: PyTorch module definitions (`SimCLR`, `ProjectorHead`, `ResNetClassifier`).
    * `train_ai.py`: SimCLR contrastive training script with NT-Xent loss and kNN monitoring logic. Contains configurable ablation tasks.
    * `train_eval.py`: Evaluation script for training a linear head on frozen representations or running supervised baselines.
    * `plot_training_data.py`: Script to generate standardized `scienceplots` graphs mapping the CSV logs to visual figures.

## 🚀 Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/charles-rl/ai-capstone-ssl.git
   cd ai-capstone-ssl
   ```

2. **Install Dependencies:**
   Make sure you have PyTorch and standard scientific libraries installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the SimCLR Model:**
   ```bash
   # Trains the model using NT-Xent loss and performs kNN monitoring
   python src/train_ai.py
   ```

4. **Evaluate the Model:**
   ```bash
   # Evaluates the frozen backbone using a Linear Probe (or other modes if configured)
   python src/train_eval.py
   ```

5. **Generate Figures:**
   ```bash
   # Reads from /dataset/training_data_csv/ to generate plots in /figures/
   python src/plot_training_data.py
   ```

---

### Author
查逸哲 Charles A. Sosmeña - National Yang Ming Chiao Tung University (NYCU).

