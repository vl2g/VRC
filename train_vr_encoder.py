import os
import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from VR_Encoder.dataloader.vrr_vg_dataset import VrRVG_train_dataset

from VR_Encoder.model.vtranse import VTransE
from VR_Encoder.model.concat import Concat

from utils.utils import set_seed, mkdir, load_config_file
from utils.logger import setup_logger

from omegaconf import OmegaConf

DATA_CONFIG_PATH = "VR_Encoder/configs/data_config.yaml"
TRAINER_CONFIG_PATH = "VR_Encoder/configs/train_config.yaml"
MODEL_CONFIG_PATH = "VR_Encoder/configs/model_config.yaml"


def save_checkpoint(config, epoch, model, optimizer):
    '''
    Checkpointing. Saves model and optimizer state_dict() and current epoch and global training steps.
    '''
    checkpoint_path = os.path.join(
        config.saved_checkpoints, f'checkpoint_{epoch}.pt')
    save_num = 0
    while (save_num < 10):
        try:

            if config.n_gpu > 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

            logger.info("Save checkpoint to {}".format(checkpoint_path))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(config, train_dataset, model):
    '''
    Trains the model.
    '''

    config.train_batch_size = config.per_gpu_train_batch_size * \
        max(1, config.n_gpu)

    # creating val set from train dataset and dataloaders
    train_size = int(config.training_split_ratio*len(train_dataset))
    val_size = len(train_dataset)-train_size

    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  shuffle=True, num_workers=0)
    val_dataloader = DataLoader(
        val_dataset, batch_size=4, shuffle=True, num_workers=0)

    # total training iterations
    t_total = len(train_dataloader) * config.num_train_epochs

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.optimizer.params.lr,
                     eps=config.optimizer.params.eps, weight_decay=config.optimizer.params.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(torch.device(config.device))

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Number of GPUs = %d", config.n_gpu)

    logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel) = %d",
                config.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    max_val_acc = 0
    epoch_val_loss_min = 1000
    val_acc_for_min_loss = 0

    for epoch in range(int(config.num_train_epochs)):
        epoch_train_loss, epoch_val_loss = 0.0, 0.0

        # train for the epoch
        model.train()
        for step, sample_batched in enumerate(train_dataloader):
            model.zero_grad()

            subj_sp = sample_batched["sub_bnd_box"].to(
                torch.device(config.device))
            obj_sp = sample_batched["obj_bnd_box"].to(
                torch.device(config.device))
            subj_cls = sample_batched["sub_class_scores"].to(
                torch.device(config.device))
            obj_cls = sample_batched["obj_class_scores"].to(
                torch.device(config.device))
            sub_feat = sample_batched["sub_roi_features"].to(
                torch.device(config.device))
            obj_feat = sample_batched["obj_roi_features"].to(
                torch.device(config.device))
            labels = sample_batched["predicate"].to(
                torch.device(config.device))

            rela_score, _ = model(
                subj_sp, subj_cls, sub_feat, obj_sp, obj_cls, obj_feat)
            loss = criterion(rela_score, labels)
            if config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # eval after the epoch
        with torch.no_grad():
            model.eval()
            num_correct, num_samples = 0, 0
            for step, sample_batched in enumerate(val_dataloader):
                subj_sp = sample_batched["sub_bnd_box"].to(
                    torch.device(config.device))
                obj_sp = sample_batched["obj_bnd_box"].to(
                    torch.device(config.device))
                subj_cls = sample_batched["sub_class_scores"].to(
                    torch.device(config.device))
                obj_cls = sample_batched["obj_class_scores"].to(
                    torch.device(config.device))
                sub_feat = sample_batched["sub_roi_features"].to(
                    torch.device(config.device))
                obj_feat = sample_batched["obj_roi_features"].to(
                    torch.device(config.device))
                labels = sample_batched["predicate"].to(
                    torch.device(config.device))

                rela_score, _ = model(
                    subj_sp, subj_cls, sub_feat, obj_sp, obj_cls, obj_feat)
                max_index = rela_score.argmax(dim=1)
                num_correct += (max_index == labels).sum()
                num_samples += labels.size(0)

                # seeing val loss
                loss = criterion(rela_score, labels)
                if config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                epoch_val_loss += loss.item()

            val_acc = (num_correct/num_samples)*100
            max_val_acc = max(val_acc, max_val_acc)
            # logger.info(f"Epoch {epoch}:Got {num_correct} / {num_samples} correct with val accuracy: {val_acc}")

            scheduler.step(val_acc)

            epoch_train_loss = epoch_train_loss / len(train_dataloader)
            epoch_val_loss = epoch_val_loss / len(val_dataloader)

            logger.info(
                f"Epoch {epoch} | Train Loss={epoch_train_loss} | Val Loss={epoch_val_loss} | Val acc. {val_acc}")

            if epoch_val_loss < epoch_val_loss_min:
                logger.info("Epoch Val loss decreased({:.6f} --> {:.6f}).Saving model ...".format(
                    epoch_val_loss_min, epoch_val_loss))
                save_checkpoint(config, epoch, model, optimizer)
                epoch_val_loss_min = epoch_val_loss
                val_acc_for_min_loss = val_acc

    return epoch_val_loss_min, val_acc_for_min_loss


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--first_argument", default=None,
                        type=str, required=False, help="some string input here")
    args = parser.parse_args()

    data_config = load_config_file(DATA_CONFIG_PATH)
    train_config = load_config_file(TRAINER_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)

    # merging data and train configs to be given to train()
    config = OmegaConf.merge(train_config, data_config)
    # config = OmegaConf.merge(OmegaConf.create(vars(args)), config) # merging cli arguments

    global logger
    # creating directories for saving checkpoints and logs
    mkdir(path=config.saved_checkpoints)
    mkdir(path=config.logs)

    logger = setup_logger(config.logs, config.logs, 0,
                          filename="training_logs.txt")

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count()  # config.n_gpu
    set_seed(seed=42, n_gpu=config.n_gpu)

    # creating model
    if model_config.model_name == "VTransE":
        model = VTransE(index_sp=model_config.index_sp,
                        index_cls=model_config.index_cls,
                        num_pred=model_config.num_pred,
                        output_size=model_config.output_size,
                        input_size=model_config.input_size)
    elif model_config.model_name == "Concat":
        model = Concat()
    else:
        logger.info(f"{model_config.model_name} model not supported")

    # getting dataset for training
    logger.info(f"Initializing dataset ...")
    train_dataset = VrRVG_train_dataset(xml_file_path=data_config.xml_file_path,
                                        npy_file_path=data_config.npy_file_path,
                                        saved_vtranse_input=data_config.saved_vtranse_input,
                                        saved_dir=data_config.saved_dir,
                                        train_predicates_path=data_config.train_predicates_path)

    # Now training
    val_loss, val_acc = train(config, train_dataset, model)

    logger.info(f"Training done: val_loss = {val_loss}, val_acc = {val_acc}")


if __name__ == "__main__":
    main()
