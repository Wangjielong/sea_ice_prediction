import datetime
import torch.cuda
from pytorchtools import EarlyStopping
import time
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

WEIGHT_FACTOR = 20


def train(epochs, patience, optimizer, model, tr_loader, val_loader, device='cpu', weight_period=None):
    """
    Train the Model using Early Stopping
    :param weight_period: must be None, melting or icing
    :param epochs:the maximum epochs used to train the network
    :param patience: How long to wait after last time validation loss improved.
    :param optimizer: the optimization method used to update the parameters of the network
    :param model: U_Net
    :param tr_loader: training dataloader
    :param val_loader: validating dataloader
    :param device: the device on which to training the network, default: 'cpu'
    :return:
    """
    print(f'Start training at {datetime.datetime.now()}')
    time_start = time.time()

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # initialize the loss function
    loss_fn = nn.MSELoss(reduction='none')

    for epoch in range(1, epochs + 1):
        # use the TensorBoard to record the loss of each sample
        tr_dir = f'tr_loss/epoch_{epoch}'
        tr_writer = SummaryWriter(log_dir=tr_dir)
        trnMetrics = torch.zeros((len(tr_loader.dataset), 1), device=device)
        tr_batch = tr_loader.batch_size

        val_dir = f'val_loss/epoch_{epoch}'
        val_writer = SummaryWriter(log_dir=val_dir)
        valMetrics = torch.zeros((len(val_loader.dataset), 1), device=device)
        val_batch = val_loader.batch_size

        # train the model
        model.train()  # prep model for training if we have BN and dropout operation
        for batch_ndx, batch_tup in enumerate(tr_loader):
            optimizer.zero_grad()  # getting rid of the gradients from the last round
            torch.cuda.empty_cache()

            tr_x = batch_tup[0]
            tr_y = batch_tup[1]

            start_ndx = batch_ndx * tr_batch
            end_ndx = start_ndx + len(tr_x)

            outputs = model(tr_x)
            sample_loss = loss_fn(outputs, tr_y)  # N*C*H*W
            loss = sample_loss.mean()

            # there is a change
            if weight_period == 'auto':
                loss = autoLoss(outputs, tr_y)
            elif weight_period:
                weighted_loss = weightedLoss(outputs, tr_y, period=weight_period)
                loss = loss + WEIGHT_FACTOR * weighted_loss

            loss.backward()
            optimizer.step()
            # record training loss
            train_losses.append(
                loss.item())  # transform the loss to a  Python number with .item(), to escape the gradients.

            trnMetrics[start_ndx:end_ndx, 0] = sample_loss.mean(dim=(1, 2, 3))

        for idx, value in enumerate(trnMetrics):
            tr_writer.add_scalar(tag='loss/train', scalar_value=value, global_step=idx)

        with torch.no_grad():
            model.eval()  # prep model for evaluation if we have BN and dropout operation
            for batch_ndx, batch_tup in enumerate(val_loader):
                val_x = batch_tup[0]
                val_y = batch_tup[1]

                start_ndx = batch_ndx * val_batch
                end_ndx = start_ndx + len(val_x)

                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(val_x)
                # calculate the loss
                sample_loss = loss_fn(outputs, val_y)
                # there is a change
                loss = sample_loss.mean()

                # there is a change
                if weight_period == 'auto':
                    loss = autoLoss(outputs, val_y)
                elif weight_period:
                    weighted_loss = weightedLoss(outputs, val_y, period=weight_period)
                    loss = loss + WEIGHT_FACTOR * weighted_loss
                # record validation loss
                valid_losses.append(loss.item())

                valMetrics[start_ndx:end_ndx, 0] = sample_loss.mean(dim=(1, 2, 3))

            for idx, value in enumerate(valMetrics):
                val_writer.add_scalar(tag='loss/val', scalar_value=value, global_step=idx)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print('{} Epoch {}, Training loss {}, Validating loss {}'.format(datetime.datetime.now(), epoch, train_loss,
                                                                         valid_loss))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop or (valid_loss < 1e-5 and train_loss < 1e-5) \
                or (epoch >= 7 and valid_loss < 1e-16):
            print("Early stopping")
            break

        tr_writer.close()
        val_writer.close()

    time_end = time.time()
    print(f'End training at {datetime.datetime.now()}')
    print(f'It takes {(time_end - time_start) / 60} minutes')

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, avg_train_losses, avg_valid_losses


def train2(epochs, patience, optimizer, model, tr_loader, val_loader, device='cpu'):
    """
    Train the Model using Early Stopping
    :param epochs:the maximum epochs used to train the network
    :param patience: How long to wait after last time validation loss improved.
    :param optimizer: the optimization method used to update the parameters of the network
    :param model: U_Net
    :param tr_loader: training dataloader
    :param val_loader: validating dataloader
    :param device: the device on which to training the network, default: 'cpu'
    :return:
    """
    print(f'Start training at {datetime.datetime.now()}')
    time_start = time.time()

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    loss_fn = nn.MSELoss(reduction='none')

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, epochs + 1):
        # use the TensorBoard to record the loss of each sample
        tr_dir = f'tr_loss/epoch_{epoch}'
        tr_writer = SummaryWriter(log_dir=tr_dir)
        trnMetrics = torch.zeros((len(tr_loader.dataset), 1), device=device)

        val_dir = f'val_loss/epoch_{epoch}'
        val_writer = SummaryWriter(log_dir=val_dir)
        valMetrics = torch.zeros((len(val_loader.dataset), 1), device=device)

        # train the model
        model.train()  # prep model for training if we have BN and dropout operation
        for batch_ndx, batch_tup in enumerate(tr_loader):
            optimizer.zero_grad()  # getting rid of the gradients from the last round
            torch.cuda.empty_cache()

            tr_x = batch_tup[0]
            tr_y = batch_tup[1]

            outputs = model(tr_x)
            loss = autoLoss(outputs, tr_y)  # N*C*H*W

            loss.backward()
            optimizer.step()
            # record training loss
            train_losses.append(
                loss.item())  # transform the loss to a  Python number with .item(), to escape the gradients.

        for idx, value in enumerate(trnMetrics):
            tr_writer.add_scalar(tag='loss/train', scalar_value=value, global_step=idx)

        with torch.no_grad():
            model.eval()  # prep model for evaluation if we have BN and dropout operation
            for batch_ndx, batch_tup in enumerate(val_loader):
                val_x = batch_tup[0]
                val_y = batch_tup[1]

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(val_x)
                # calculate the loss
                sample_loss = loss_fn(output, val_y)
                # there is a change
                loss = sample_loss.mean()
                # there is a change

                # record validation loss
                valid_losses.append(loss.item())

            for idx, value in enumerate(valMetrics):
                val_writer.add_scalar(tag='loss/val', scalar_value=value, global_step=idx)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print('{} Epoch {}, Training loss {}, Validating loss {}'.format(datetime.datetime.now(), epoch, train_loss,
                                                                         valid_loss))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop or (valid_loss < 1e-5 and train_loss < 1e-5) \
                or (epoch >= 7 and valid_loss < 1e-16):
            print("Early stopping")
            break

        tr_writer.close()
        val_writer.close()

    time_end = time.time()
    print(f'End training at {datetime.datetime.now()}')
    print(f'It takes {(time_end - time_start) / 60} minutes')

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, avg_train_losses, avg_valid_losses


def weightedLoss(prediction, target, period='melting'):
    """

    :param prediction: the prediction from our model
    :param target: the target data
    :param period: the flag of icing and melting, default melting
    :return:
    """
    if period == 'melting':
        threshold = 0.5
        target_bool = target < threshold
    elif period == 'icing':
        threshold = 0.6
        target_bool = target > threshold
    elif period == 'existence':
        threshold = 0.0
        target_bool = target > threshold
    else:
        raise ValueError('must be icing, melting or existence')

    prediction_weighted = prediction * target_bool
    target_weighted = target * target_bool
    loss_fn = nn.MSELoss()
    weighted_loss = loss_fn(prediction_weighted, target_weighted)

    return weighted_loss


def autoLoss(prediction, target):
    """
    automatically discern the period of melting or icing to calculate the loss
    :param prediction: the prediction from our model
    :param target: the target data
    :return:
    """

    loss_fn = nn.MSELoss()

    loss = loss_fn(prediction[target != 0], target[target != 0])

    return loss
