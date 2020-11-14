import torch
from dataset import TweetDataset, collate_function
from tensorboardX import SummaryWriter
from torch.optim import Adam
from models import RNN, Classifier
from config import TRAIN_CONFIG
from modules import printProgressBar, AverageMeter, MAE, MSE, precision_recall
from torch.utils.data import DataLoader
import time
import datetime
import os
from torch.nn import BCELoss, L1Loss

cfg = TRAIN_CONFIG


def infer_RNN(model, batch, loss_fun):
    # inference
    target = batch['target'].cuda().unsqueeze(1)
    numeric = batch['numeric'].cuda()
    model_input = batch['embedding'].cuda()
    model_output = model(model_input, numeric)

    # loss
    return loss_fun(model_output, target)  # still validating on MAE


def infer_Classifier(model, batch, loss_fun):
    t = 1
    target = batch['target'].unsqueeze(1)
    target[target < t] = 0.
    target[target >= t] = 1.
    target = target.cuda()
    numeric = batch['numeric'].cuda()
    model_input = batch['embedding'].cuda()
    model_output = model(model_input, numeric)

    return loss_fun(model_output.float(), target.float())


def val(model, val_loader, writer, step, infer, loss_fun):
    """
    Computes the loss on the validation set and logs it to tensorboard \n
    The loss is computed on a fixed subset with the first [val_batches] batches, defined in config file
    """
    print('\n')
    model = model.eval()
    with torch.no_grad():
        total_val_loss = 0.
        for batch_idx, batch in enumerate(val_loader):

            # run only on a subset
            if batch_idx >= cfg['val_batches']:
                break

            # loss = infer_RNN(model, batch, MAE)
            loss = infer(model, batch, loss_fun)

            # log
            printProgressBar(batch_idx, cfg['val_batches'], suffix='\tValidation ...')

            total_val_loss += loss

    val_loss = total_val_loss / cfg['val_batches']
    writer.add_scalar('Steps/val_loss', val_loss, step)
    print('\n')
    print('Finished validation with loss {:4f}'.format(val_loss))


def train(model, infer, train_loss_fun, val_loss_fun, load_checkpoint=None):
    """ Train the RNN model using the parameters defined in the config file """
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
    print('Starting training')
    for epoch in range(start_epoch, epochs):
        loader_length = len(train_loader)
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimiser.zero_grad()

            loss = infer(model, batch, train_loss_fun)
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
                             prefix='Epoch [{}/{}]\tStep [{}/{}]'.format(epoch, epochs-1, batch_idx, loader_length))

            writer.add_scalar('Steps/train_loss', loss, step)

            # saving the model
            if step % cfg['checkpoint_every'] == 0:
                checkpoint_name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step,
                            'optimiser': optimiser.state_dict(), 'train_config': cfg, 'net_config': model.config},
                           checkpoint_name)

            # validating
            if step % cfg['val_every'] == 0:
                val(model, val_loader, writer, step, infer, val_loss_fun)
                model = model.train()

            step += 1
            optimiser.step()

        # end of epoch
        print('')
        writer.add_scalar('Epochs/train_loss', avg_loss.avg, epoch)
        avg_loss.reset()
        checkpoint_name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step,
                    'optimiser': optimiser.state_dict(), 'train_config': cfg, 'net_config': model.config},
                   checkpoint_name)

    # finished training
    writer.close()
    print('Training finished :)')


if __name__ == '__main__':
    net = RNN().train().cuda()
    train(net, infer_Classifier, BCELoss(), BCELoss())  #, load_checkpoint='checkpoints/test_new_rnn_3/epoch_9.pth')
