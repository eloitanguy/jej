import torch
from dataset import TweetDataset, collate_function
from tensorboardX import SummaryWriter
from torch.optim import Adam
from models import RNN, CNN
from config import TRAIN_CONFIG, DATASET_CONFIG
from modules import printProgressBar, AverageMeter
from torch.utils.data import DataLoader
import time
import datetime
import os
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss

cfg = TRAIN_CONFIG


def infer_NN(model, batch):
    loss_fun = L1Loss()
    # inference
    target = batch['target'].cuda().unsqueeze(1).float()
    numeric = batch['numeric'].cuda()
    model_input = batch['embedding'].cuda()
    model_output = model(model_input, numeric)

    # loss
    return loss_fun(model_output, target)


def infer_logNN(model, batch):
    loss_fun = MSELoss()
    # inference
    target = batch['target'].cuda().unsqueeze(1).float()
    numeric = batch['numeric'].cuda()
    model_input = batch['embedding'].cuda()
    model_output = model(model_input, numeric)

    # loss
    return loss_fun(model_output, torch.log(1 + target))  # Add 5 for centering on e^5


def eval_logNN(model, batch):
    loss_fun = L1Loss()
    # inference
    target = batch['target'].cuda().squeeze().float()
    numeric = batch['numeric'].cuda()
    model_input = batch['embedding'].cuda()
    model_output = model(model_input, numeric).squeeze()

    # loss
    return loss_fun(torch.exp(model_output) - 1, target)


def infer_Classifier(model, batch):
    loss_fun = BCEWithLogitsLoss()
    t = 1
    target = batch['target'].unsqueeze(1)
    target[target < t] = 0.
    target[target >= t] = 1.
    target = target.cuda()
    numeric = batch['numeric'].cuda()
    model_input = batch['embedding'].cuda()
    model_output, _ = model(model_input, numeric)

    return loss_fun(model_output.float(), target.float())


def val(model, val_loader, writer, step, infer):
    """
    Computes the loss on the validation set and logs it to tensorboard \n
    The loss is computed on a fixed subset with the first [val_batches] batches, defined in config file
    """
    print('\n')
    model.eval()
    val_losses = []
    n = len(val_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):

            # run only on a subset
            if batch_idx >= cfg['val_batches']:
                break

            batch_val_loss = infer(model, batch).item()

            # log
            printProgressBar(batch_idx, min(n, cfg['val_batches']), suffix='\tValidation ...')

            val_losses.append(batch_val_loss)

        val_loss = sum(val_losses) / len(val_losses)
    writer.add_scalar('Steps/val_loss', val_loss, step)
    print('\n')
    print('Finished validation with loss {:4f}'.format(val_loss))
    return val_loss


def train(model, infer_train, infer_val, load_checkpoint=None):
    """ Train the RNN model using the parameters defined in the config file """
    global checkpoint_name
    print('Initialising {}'.format(cfg['experiment_name']))
    checkpoint_folder = 'checkpoints/{}/'.format(cfg['experiment_name'])

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    tb_folder = 'tb/{}/'.format(cfg['experiment_name'])
    if not os.path.exists(tb_folder):
        os.makedirs(tb_folder)

    writer = SummaryWriter(logdir=tb_folder, flush_secs=30)
    optimiser = Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    train_dataset = TweetDataset(dataset_type='train')
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'],
                              collate_fn=collate_function, shuffle=True, pin_memory=True)

    val_dataset = TweetDataset(dataset_type='val')
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'],
                            collate_fn=collate_function, shuffle=False, pin_memory=True)

    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint)
        assert model.config == checkpoint['net_config'], \
            "The provided checkpoint has a different configuration, loading is impossible"
        start_epoch = checkpoint['epoch'] + 1
        epochs = cfg['epochs'] + start_epoch
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model'])
        optimiser.load_state_dict(checkpoint['optimiser'])
        print("Loaded the checkpoint at {}".format(load_checkpoint))
    else:
        start_epoch, step = 0, 0
        epochs = cfg['epochs']

    init_loss = 0.
    avg_loss = AverageMeter()
    best_mae = 1e10

    print('Sanity val')
    val(model, val_loader, writer, 0, infer_val)
    model.train()

    print('Starting training')
    for epoch in range(start_epoch, epochs):
        loader_length = len(train_loader)
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimiser.zero_grad()

            loss = infer_train(model, batch)
            loss.backward()

            if epoch == 0 and batch_idx == 0:
                init_loss = loss

            # logging
            elapsed = time.time() - epoch_start
            progress = batch_idx / loader_length
            est = datetime.timedelta(seconds=int(elapsed / progress)) if progress > 0.001 else '-'
            avg_loss.update(loss)
            suffix = '\tloss {:.4f}/{:.4f}\tETA [{}/{}]'.format(avg_loss.avg, init_loss,
                                                                datetime.timedelta(seconds=int(elapsed)), est)
            printProgressBar(batch_idx, loader_length, suffix=suffix,
                             prefix='Epoch [{}/{}]\tStep [{}/{}]'.format(epoch, epochs - 1, batch_idx, loader_length))

            writer.add_scalar('Steps/train_loss', loss, step)

            # saving the model
            if step % cfg['checkpoint_every'] == 0:
                checkpoint_name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': batch_idx, 'step': step,
                            'optimiser': optimiser.state_dict(), 'train_config': cfg, 'net_config': model.config,
                            'dataset_config': DATASET_CONFIG},
                           checkpoint_name)
            step += 1
            optimiser.step()

            # validating
            if step % cfg['val_every'] == 0:
                mae = val(model, val_loader, writer, step, infer_val)
                if mae < best_mae:
                    best_mae = mae
                    print('Best model with V{:.2f}'.format(best_mae))
                    torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': batch_idx, 'step': step,
                                'optimiser': optimiser.state_dict(), 'train_config': cfg, 'net_config': model.config,
                                'dataset_config': DATASET_CONFIG},
                               '{}/best_with_V{:.2f}.pth'.format(checkpoint_folder, best_mae))
                model.train()

        # end of epoch
        print('')
        writer.add_scalar('Epochs/train_loss', avg_loss.avg, epoch)
        avg_loss.reset()
        checkpoint_name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step,
                    'optimiser': optimiser.state_dict(), 'train_config': cfg, 'net_config': model.config,
                    'dataset_config': DATASET_CONFIG},
                   checkpoint_name)

    # finished training
    writer.close()
    print('Training finished :)')


if __name__ == '__main__':
    net = RNN().train().cuda()
    train(net, infer_logNN, eval_logNN)
