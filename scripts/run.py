import torch
import torch.nn as nn
from loader import get_dataloaders
from models import get_model

import numpy as np
import argparse
import datetime
from torch.utils.tensorboard import SummaryWriter

import optuna
import logging
import sys
import random

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler("output.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger()

# train one epoch
def train(train_loader, model, loss_fn, optimizer, logger, writer):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Train
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        y = y.unsqueeze(1).float()
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(datetime.datetime.now())
        # Show progress
        los = loss.item()
        writer.add_scalar('loss/train', los, batch)
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"train loss: {loss:>7f} [{current:>5d}/{len(train_loader.dataset):>5d}]")


# validate and return mae loss
def validate(val_loader, model, logger):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Validation
    model.eval()
    val_loss_mse = 0
    val_loss_mae = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            y = y.unsqueeze(1)

            loss_mse = nn.MSELoss()(pred, y)
            val_loss_mse += loss_mse.item()
            loss_mae = nn.L1Loss()(pred, y)
            val_loss_mae += loss_mae.item()

    val_loss_mse /= len(val_loader)
    val_loss_mae /= len(val_loader)

    logger.info(f"val mse loss: {val_loss_mse:>7f}, val mae loss: {val_loss_mae}")
    return val_loss_mae



# test and return mse and mae loss
def test(test_loader, model, logger):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Test
    model.eval()
    test_loss_mse = 0
    test_loss_mae = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            y = y.unsqueeze(1)

            loss_mse = nn.MSELoss()(pred, y)
            test_loss_mse += loss_mse.item()
            loss_mae = nn.L1Loss()(pred, y)
            test_loss_mae += loss_mae.item()

    test_loss_mse /= len(test_loader)
    test_loss_mae /= len(test_loader)

    logger.info(f"test mse loss: {test_loss_mse:>7f}, test mae loss: {test_loss_mae}")
    return test_loss_mse, test_loss_mae



# helper class for early stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, logger, s):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger, s)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger, s)
            self.counter = 0
        
        return self.val_loss_min

    def save_checkpoint(self, val_loss, model, logger, s):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'../weights/{s}.pt')  # save checkpoint
        self.val_loss_min = val_loss



def objective(trial: optuna.Trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    bs = trial.suggest_categorical("batch_size", [64, 128])

    n_layers = trial.suggest_int('n_layers', 0, 3)
    optimize = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    non_linearity = trial.suggest_categorical('non-linearity', ["gelu", "swish"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Using " + device + "...")
    s = f"lr{lr:.5f}-bs{bs}-hidden{n_layers}-o{optimize}-n{non_linearity}_best"

    writer = SummaryWriter(f'runs/{s}')
    layers = []
    in_features = 768
    config = []
    train_loader, val_loader, test_loader = get_dataloaders(bs, augmented=args.augmented, vit_transformed=True, show_sample=True)

    if n_layers != 0:
        for i in range(n_layers):
            out = trial.suggest_int(f'n_units_l{i}', 80, in_features)
            layers.append(torch.nn.Linear(in_features, out))
            config.append(f"Linear({in_features}, {out})")
            layers.append(torch.nn.GELU() if non_linearity == "gelu" else torch.nn.SiLU())
            config.append(f"{non_linearity.upper()}")
            if 1 < n_layers and i < n_layers-1:
                layers.append(torch.nn.Dropout(0.5))
                config.append("Dropout(0.5)")
            in_features = out

    layers.append(torch.nn.Linear(in_features, 1))
    config.append(f"Linear({in_features}, 1)")

    model_head = torch.nn.Sequential(*layers).to(device)
    logger.info(f"MLP Configuration: {' -> '.join(config)}")

    model = get_model()
    model.heads = model_head
    model = model.float().to(device)

    loss_fn = nn.MSELoss() # TODO: focal loss

    if optimize == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    epochs = 50
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for t in range(epochs):
        logger.info(f"Epoch {t + 1}\n-------------------------------")
        logger.info(f"learning rate: {optimizer.param_groups[0]['lr']}")
        logger.info(f"batch size: {bs}")
        train(train_loader, model, loss_fn, optimizer, logger, writer)
        val_loss = validate(test_loader, model, logger)
        writer.add_scalar('mae_loss/validation', val_loss, t)
        min_loss = early_stopping(val_loss, model, logger, s)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        scheduler.step()

    writer.flush()
    writer.close()
    model.load_state_dict(torch.load(f'../weights/{s}.pt'))
    test(test_loader, model, logger)
    
    logger.info("Done!")
    return min_loss



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmented', type=bool, default=False, help='set to True to use augmented dataset')
    parser.add_argument('--optuna', type=bool, default=False, help='Run Optuna study')
    parser.add_argument('--trials', type=int, default=30, help='Number of trials for Optuna study')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of hidden layers')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type (adam or adamw)')
    parser.add_argument('--non_linearity', type=str, default='gelu', help='Non-linearity function (gelu or swish)')
    parser.add_argument('--early_stop_threshold', type=int, default=5, help='Early stopping patience threshold')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='Scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.3, help='Scheduler gamma value')
    parser.add_argument('--dataset_fraction', type=float, default=1.0, help='Fraction of dataset that is used in training')
    args = parser.parse_args()

    if args.optuna_study:
        study = optuna.create_study()
        study.optimize(objective, n_trials=args.trials)
        logger.info(f"Best Params: {study.best_params}")
    else:
        lr = args.lr
        bs = args.bs
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Using " + device + "...")
        s = f"lr{args.lr:.5f}-bs{args.bs}-hidden{args.n_layers}-o{args.optimizer}-n{args.non_linearity}_best"

        writer = SummaryWriter(f'runs/{s}')
        layers = []
        in_features = 768
        config = []
        train_loader, val_loader, test_loader = get_dataloaders(args.bs, augmented=args.augmented, vit_transformed=True, show_sample=True, fraction=args.fraction)
        
        if n_layers != 0:
            for i in range(n_layers):
                out = random.randint(80, in_features)
                layers.append(torch.nn.Linear(in_features, out))
                config.append(f"Linear({in_features}, {out})")
                layers.append(torch.nn.GELU() if args.non_linearity == "gelu" else torch.nn.SiLU())
                config.append(f"{args.non_linearity.upper()}")
                if 1 < n_layers and i < n_layers-1:
                    layers.append(torch.nn.Dropout(0.5))
                    config.append("Dropout(0.5)")
                in_features = out
            
        layers.append(torch.nn.Linear(in_features, 1))
        config.append(f"Linear({in_features}, 1)")

        model_head = torch.nn.Sequential(*layers).to(device)
        logger.info(f"MLP Configuration: {' -> '.join(config)}")

        model = get_model()
        model.heads = model_head
        model = model.float().to(device)

        loss_fn = nn.MSELoss() # TODO: focal loss


        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        epochs = args.epochs
        early_stopping = EarlyStopping(patience=args.early_stop_threshold, verbose=True)

        for t in range(epochs):
            logger.info(f"Epoch {t + 1}\n-------------------------------")
            logger.info(f"learning rate: {optimizer.param_groups[0]['lr']}")
            train(train_loader, model, loss_fn, optimizer, logger, writer)
            val_loss = validate(test_loader, model, logger)
            writer.add_scalar('mae_loss/validation', val_loss, t)
            min_loss = early_stopping(val_loss, model, logger, s)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
            scheduler.step()

        writer.flush()
        writer.close()
        model.load_state_dict(torch.load(f'../weights/{s}.pt'))
        test(test_loader, model, logger)
        
        logger.info("Done!")

