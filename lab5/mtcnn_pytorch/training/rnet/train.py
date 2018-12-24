import sys
sys.path.append('./')

import argparse
import torch
from tools.imagedb import FaceDataset
from torchvision import transforms
from models.rnet import RNet
from training.rnet.trainer import RNetTrainer
from training.rnet.config import Config
# from tools.logger import Logger
from checkpoint import CheckPoint
import os
import config

# Get config
config = Config()
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)

# Set device
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
use_cuda = config.use_cuda and torch.cuda.is_available()
torch.manual_seed(config.manualSeed)
torch.cuda.manual_seed(config.manualSeed)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Set dataloader
kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_loader = torch.utils.data.DataLoader(
    FaceDataset(config.annoPath, transform=transform, is_train=True), batch_size=config.batchSize, shuffle=True, **kwargs)

# Set model
model = RNet()
model = model.to(device)

# Set checkpoint
checkpoint = CheckPoint(config.save_path)

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

# Set trainer
# logger = Logger(config.save_path)
logger = None
trainer = RNetTrainer(config.lr, train_loader, model, optimizer, scheduler, logger, device)

for epoch in range(1, config.nEpochs + 1):
    cls_loss_, box_offset_loss, total_loss, accuracy = trainer.train(epoch)
    checkpoint.save_model(model, index=epoch)
