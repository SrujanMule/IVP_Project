#Interface

from model import EfficientNetCustom as TheModel
from train import train_model as the_trainer
from predict import classify_galaxy as the_predictor
from dataset import GalaxyZooDataset as TheDataset
from dataset import GalaxyZooDataloader as the_dataloader
from config import batchsize as the_batch_size
from config import epochs as total_epochs
