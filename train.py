import torch
from dataset import TweetDataset, collate_function
from tensorboardX import SummaryWriter
from torch.optim import Adam
from models import RNN
from config import TRAIN_CONFIG
from modules import printProgressBar, AverageMeter, MAE, MSE
from torch.utils.data import DataLoader
import time
import datetime
import os

cfg = TRAIN_CONFIG


def val_RNN(model, val_loader, writer, step):
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

            # inference
            target = batch['target'].cuda()
            model_input = batch['embedding'].cuda()
            model_output = model(model_input)

            # loss
            loss = MAE(target, model_output)  # still validating on MAE

            # log
            printProgressBar(batch_idx, cfg['val_batches'], suffix='\tValidation ...')

            total_val_loss += loss

    val_loss = total_val_loss / cfg['val_batches']
    writer.add_scalar('Steps/val_loss', val_loss, step)
    print('\n')
    print('Finished validation with loss {:4f}'.format(val_loss))


def train_RNN():
    """ Train the RNN model using the parameters defined in the config file """
    print('Initialising ...')
    checkpoint_folder = 'checkpoints/{}/'.format(cfg['experiment_name'])

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    tb_folder = 'tb/{}/'.format(cfg['experiment_name'])
    if not os.path.exists(tb_folder):
        os.makedirs(tb_folder)

    writer = SummaryWriter(logdir=tb_folder, flush_secs=30)
    model = RNN(hidden_size=cfg['RNN_hidden_units']).cuda().train()
    optimiser = Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    train_dataset = TweetDataset(dataset_type='train')
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'],
                              collate_fn=collate_function, pin_memory=True)

    val_dataset = TweetDataset(dataset_type='val')
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'],
                            collate_fn=collate_function, shuffle=False, pin_memory=True)

    epochs = cfg['epochs']
    init_loss, step = 0., 0
    avg_loss = AverageMeter()
    print('Starting training')
    for epoch in range(epochs):
        loader_length = len(train_loader)
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimiser.zero_grad()

            # inference
            target = batch['target'].cuda()
            model_input = batch['embedding'].cuda()
            model_output = model(model_input)

            # loss
            loss = MSE(target, model_output)  # testing MSE and not MAE
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
                             prefix='Epoch [{}/{}]\tStep [{}/{}]'.format(epoch, epochs, batch_idx, loader_length))

            writer.add_scalar('Steps/train_loss', loss, step)

            # saving the model
            if step % cfg['checkpoint_every'] == 0:
                checkpoint_name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step,
                            'optimiser': optimiser.state_dict(), 'config': cfg}, checkpoint_name)

            # validating
            if step % cfg['val_every'] == 0:
                val_RNN(model, val_loader, writer, step)
                model = model.train()

            step += 1
            optimiser.step()

        # end of epoch
        print('')
        writer.add_scalar('Epochs/train_loss', avg_loss.avg, epoch)
        avg_loss.reset()
        checkpoint_name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step,
                    'optimiser': optimiser.state_dict(), 'config': cfg}, checkpoint_name)

    # finished training
    writer.close()
    print('Training finished :)')


if __name__ == '__main__':
    val_dataset = TweetDataset(dataset_type='val')
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'],
                            collate_fn=collate_function, shuffle=False, pin_memory=True)
    checkpoint = torch.load("checkpoints/test/epoch_3.pth")
    model = RNN(hidden_size=cfg['RNN_hidden_units'])
    model.load_state_dict(checkpoint['model'])
    model = model.eval().cuda()

    for batch in val_loader:
        target = batch['target'].cuda()
        model_input = batch['embedding'].cuda()
        model_output = model(model_input)
        print(model_output[:100])
        break
    # train_RNN()
