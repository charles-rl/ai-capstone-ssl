import torch
import torchvision.transforms as T
import wandb
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from training_models import ResNetClassifier


# --- HYPERPARAMETERS (Per PDF Instructions) ---
EPOCHS = 100
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SSL checkpoint used in linear probe mode
SIMCLR_CHKPT_PATH = "./models/best_simclr_model.pth"

CONFIG = {
	"learning_rate": 1e-3,
	"weight_decay": 1e-6,
	"batch_size": BATCH_SIZE,
	"epochs": EPOCHS,
	"mode": "linear_probe",  # "linear_probe", "supervised", "random_init"
}


class SupervisedTransform:
	def __init__(self, transform_pipeline):
		self.transform_pipeline = transform_pipeline

	def __call__(self, x):
		return self.transform_pipeline(x)


# CIFAR-10 standard mean and std
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

train_transforms = SupervisedTransform(
	T.Compose(
		[
			T.RandomCrop(32, padding=4),
			T.RandomHorizontalFlip(),
			T.ToTensor(),
			T.Normalize(cifar10_mean, cifar10_std),
		]
	)
)

test_transforms = SupervisedTransform(
	T.Compose(
		[
			T.ToTensor(),
			T.Normalize(cifar10_mean, cifar10_std),
		]
	)
)


@torch.no_grad()
def evaluate(model, data_loader):
	model.eval()
	correct = 0
	total = 0

	for x, y in data_loader:
		x, y = x.to(DEVICE), y.to(DEVICE)
		logits = model(x)
		preds = torch.argmax(logits, dim=1)

		correct += (preds == y).sum().item()
		total += y.size(0)

	return (correct / total) * 100.0


def train():
	run_name = f"Eval-{CONFIG['mode']}"
	wandb.init(project="NYCU-AI-Capstone-Project2", config=CONFIG, name=run_name)
	print(f"Device: {DEVICE}")

	train_dataset = CIFAR10(
		root="./dataset", train=True, download=True, transform=train_transforms
	)
	train_loader = DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=4,
		drop_last=True,
	)

	test_dataset = CIFAR10(
		root="./dataset", train=False, download=True, transform=test_transforms
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=4,
	)

	model = ResNetClassifier(
		config=CONFIG,
		simclr_chkpt_path=SIMCLR_CHKPT_PATH,
		device=DEVICE,
	)

	best_test_acc = 0.0

	for epoch in range(EPOCHS):
		model.train()
		epoch_loss = 0.0
		epoch_acc = 0.0

		for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
			loss, acc = model.learn(batch_x, batch_y)
			epoch_loss += loss
			epoch_acc += acc

		avg_train_loss = epoch_loss / len(train_loader)
		avg_train_acc = (epoch_acc / len(train_loader)) * 100.0

		test_acc = evaluate(model, test_loader)
		best_test_acc = max(best_test_acc, test_acc)

		print(
			f"Epoch {epoch + 1:03d} | "
			f"Train Loss: {avg_train_loss:.4f} | "
			f"Train Acc: {avg_train_acc:.2f}% | "
			f"Test Acc: {test_acc:.2f}%"
		)

		wandb.log(
			{
				"epoch": epoch + 1,
				"Train Loss": avg_train_loss,
				"Train Acc": avg_train_acc,
				"Test Acc": test_acc,
			}
		)

	wandb.run.summary["Best Test Acc"] = best_test_acc
	wandb.finish()


if __name__ == "__main__":
	train()
