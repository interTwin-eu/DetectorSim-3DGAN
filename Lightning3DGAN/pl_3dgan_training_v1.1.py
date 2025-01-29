import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import h5py
import numpy as np
from collections import defaultdict
import pytorch_lightning as pl
import sys
import pickle
import time

from typing import Optional, Tuple, Dict, Any
import os
#from lightning.pytorch.utilities.types import EVAL_DATALOADERS

import wandb
from pytorch_lightning.loggers import WandbLogger

from pl_3dgan_models_v1 import *
from analysis_utils import *


class ParticlesDataset(Dataset):
    def __init__(self, datapath="/eos/user/k/ktsolaki/data/3dgan_data/*.h5", max_samples: Optional[int] = None): #/eos/user/k/ktsolaki/data/3dgan_data/*.h5 afs/cern.ch/work/k/ktsolaki/private/projects/GAN_scripts/3DGAN/Accelerated3DGAN/src/Accelerated3DGAN/data/*.h5
        self.datapath = datapath
        self.max_samples = max_samples
        self.data = dict()

        self.fetch_data()

    def __len__(self):
        return len(self.data["X"])

    def __getitem__(self, idx):
        return {"X": self.data["X"][idx], "Y": self.data["Y"][idx], "ang": self.data["ang"][idx], "ecal": self.data["ecal"][idx]}

    def fetch_data(self) -> None:

        print("Searching in :", self.datapath)
        files = sorted(glob.glob(self.datapath))
        print("Found {} files. ".format(len(files)))
        if len(files) == 0:
            raise RuntimeError(f"No H5 files found at '{self.datapath}'!")

        # concatenated_datasets = []
        # for datafile in Files:
        #   f=h5py.File(datafile,'r')
        #   dataset = self.GetDataAngleParallel(f)
        #   concatenated_datasets.append(dataset)
        #   result = {key: [] for key in concatenated_datasets[0].keys()} # Initialize result dictionary
        #   for d in concatenated_datasets:
        #     for key in result.keys():
        #       result[key].extend(d[key])
        # return result

        for datafile in files:
            f = h5py.File(datafile, 'r')
            dataset = self.GetDataAngleParallel(f)
            for field, vals_array in dataset.items():
                # Cast to float32 the full energy data (if using restricted energy data comment out the following line)
                vals_array = vals_array.astype(np.float32)
                if self.data.get(field) is not None:
                    # Resize to include the new array
                    new_shape = list(self.data[field].shape)
                    new_shape[0] += len(vals_array)
                    self.data[field].resize(new_shape)
                    self.data[field][-len(vals_array):] = vals_array
                else:
                    self.data[field] = vals_array

            # Stop loading data, if self.max_samples reached
            if (self.max_samples is not None
                    and len(self.data[field]) >= self.max_samples):
                for field, vals_array in self.data.items():
                    self.data[field] = vals_array[:self.max_samples]

                break

    def GetDataAngleParallel(
    self,
    dataset,
    xscale=1,
    xpower=0.85,
    yscale=100,
    angscale=1,
    angtype="theta",
    thresh=1e-4,
    daxis=-1,):
      """Preprocess function for the dataset

      Args:
          dataset (str): Dataset file path
          xscale (int, optional): Value to scale the ECAL values. Defaults to 1.
          xpower (int, optional): Value to scale the ECAL values, exponentially. Defaults to 1.
          yscale (int, optional): Value to scale the energy values. Defaults to 100.
          angscale (int, optional): Value to scale the angle values. Defaults to 1.
          angtype (str, optional): Which type of angle to use. Defaults to "theta".
          thresh (_type_, optional): Maximum value for ECAL values. Defaults to 1e-4.
          daxis (int, optional): Axis to expand values. Defaults to -1.

      Returns:
        Dict: Dictionary containning the preprocessed dataset
      """
      X = np.array(dataset.get("ECAL")) * xscale
      Y = np.array(dataset.get("energy")) / yscale
      X[X < thresh] = 0
      X = X.astype(np.float32)
      Y = Y.astype(np.float32)
      ecal = np.sum(X, axis=(1, 2, 3))
      indexes = np.where(ecal > 10.0)
      X = X[indexes]
      Y = Y[indexes]
      if angtype in dataset:
          ang = np.array(dataset.get(angtype))[indexes]
      # else:
      # ang = gan.measPython(X)
      X = np.expand_dims(X, axis=daxis)
      ecal = ecal[indexes]
      ecal = np.expand_dims(ecal, axis=daxis)
      if xpower != 1.0:
          X = np.power(X, xpower)

      Y = np.array([[el] for el in Y])
      ang = np.array([[el] for el in ang])
      ecal = np.array([[el] for el in ecal])

      final_dataset = {"X": X, "Y": Y, "ang": ang, "ecal": ecal}

      return final_dataset


class ParticlesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, datapath="/eos/user/k/ktsolaki/data/3dgan_data/*.h5", num_workers: int = 4, max_samples: Optional[int] = None) -> None: #/eos/user/k/ktsolaki/data/3dgan_data/*.h5 afs/cern.ch/work/k/ktsolaki/private/projects/GAN_scripts/3DGAN/Accelerated3DGAN/src/Accelerated3DGAN/data/*.h5
        super().__init__()
        self.batch_size = batch_size
        self.datapath = datapath
        self.num_workers = num_workers
        self.max_samples = max_samples

    def setup(self, stage: str = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        if stage == 'fit' or stage is None:
            self.dataset = ParticlesDataset(self.datapath, max_samples=self.max_samples)
            dataset_length = len(self.dataset)
            split_point = int(dataset_length * 0.9)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [split_point, dataset_length - split_point])

        if stage == 'predict':
            # TODO: inference dataset should be different in that it
            # does not contain images!
            self.predict_dataset = ParticlesDataset(
                self.datapath,
                max_samples=self.max_samples
            )

        #if stage == 'test' or stage is None:
            #self.test_dataset = MyDataset(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers)

    def predict_dataloader(self): #-> EVAL_DATALOADERS
        return DataLoader(self.predict_dataset, num_workers=self.num_workers, batch_size=self.batch_size, drop_last=True) 

    #def test_dataloader(self):
        #return DataLoader(self.test_dataset, batch_size=self.batch_size)


class ThreeDGAN(pl.LightningModule):
    def __init__(self, latent_size=256, batch_size=64, loss_weights=[3, 0.1, 25, 0.1], power=0.85, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.latent_size = latent_size
        self.batch_size = batch_size
        self.loss_weights = loss_weights
        self.lr = lr
        self.power = power

        self.generator = Generator(self.hparams.latent_size)
        self.discriminator = Discriminator(self.hparams.power)

        self.epoch_gen_loss = []
        self.epoch_disc_loss = []
        self.disc_epoch_test_loss = []
        self.gen_epoch_test_loss = []
        self.index = 0
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.pklfile = "/eos/user/k/ktsolaki/misc/3dgan_pytorch/3dgan_history_30ep_all.pkl"


    def BitFlip(self, x, prob=0.05):
        """
        Flips a single bit according to a certain probability.

        Args:
            x (list): list of bits to be flipped
            prob (float): probability of flipping one bit

        Returns:
            list: List of flipped bits

        """
        x = np.array(x)
        selection = np.random.uniform(0, 1, x.shape) < prob
        x[selection] = 1 * np.logical_not(x[selection])
        return x

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-7))) * 100

    def compute_global_loss(self, labels, predictions, loss_weights=[3, 0.1, 25, 0.1]):
        # Can be initialized outside
        binary_crossentropy_object = nn.BCELoss(reduction='none')
        #there is no equivalent in pytorch for tf.keras.losses.MeanAbsolutePercentageError --> using the custom "mean_absolute_percentage_error" above!
        mean_absolute_percentage_error_object1 = self.mean_absolute_percentage_error(predictions[1], labels[1])
        mean_absolute_percentage_error_object2 = self.mean_absolute_percentage_error(predictions[3], labels[3])
        mae_object = nn.L1Loss(reduction='none')

        binary_example_loss = binary_crossentropy_object(predictions[0], labels[0]) * loss_weights[0]

        #mean_example_loss_1 = mean_absolute_percentage_error_object(predictions[1], labels[1]) * loss_weights[1]
        mean_example_loss_1 = mean_absolute_percentage_error_object1 * loss_weights[1]

        mae_example_loss = mae_object(predictions[2], labels[2]) * loss_weights[2]

        #mean_example_loss_2 = mean_absolute_percentage_error_object(predictions[3], labels[3]) * loss_weights[3]
        mean_example_loss_2 = mean_absolute_percentage_error_object2 * loss_weights[3]

        binary_loss = binary_example_loss.mean()
        mean_loss_1 = mean_example_loss_1.mean()
        mae_loss = mae_example_loss.mean()
        mean_loss_2 = mean_example_loss_2.mean()

        return [binary_loss, mean_loss_1, mae_loss, mean_loss_2]

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        image_batch, energy_batch, ang_batch, ecal_batch = batch['X'], batch['Y'], batch['ang'], batch['ecal']

        image_batch = image_batch.permute(0, 4, 1, 2, 3)

        image_batch = image_batch.to(self.device)
        energy_batch = energy_batch.to(self.device)
        ang_batch = ang_batch.to(self.device)
        ecal_batch = ecal_batch.to(self.device)

        optimizer_discriminator, optimizer_generator = self.optimizers()

        noise = torch.randn((self.batch_size, self.latent_size - 2)).to(self.device)
        generator_ip = torch.cat(
            (energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise),
            dim=1,)
        generated_images = self.generator(generator_ip)

        # Train discriminator first on real batch
        fake_batch = self.BitFlip(np.ones(self.batch_size).astype(np.float32))
        fake_batch = torch.tensor([[el] for el in fake_batch]).to(self.device)
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        predictions = self.discriminator(image_batch)
        print("calculating real_batch_loss...")
        real_batch_loss = self.compute_global_loss(
            labels, predictions, self.loss_weights)
        self.log("real_batch_loss", sum(real_batch_loss), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        print("real batch disc train")
        #the following 3 lines correspond in tf version to:
        #gradients = tape.gradient(real_batch_loss, discriminator.trainable_variables)
        #optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) in Tensorflow
        optimizer_discriminator.zero_grad()
        self.manual_backward(sum(real_batch_loss))
        #sum(real_batch_loss).backward()
        #real_batch_loss.backward()
        optimizer_discriminator.step()

        # Train discriminator on the fake batch
        fake_batch = self.BitFlip(np.zeros(self.batch_size).astype(np.float32))
        fake_batch = torch.tensor([[el] for el in fake_batch]).to(self.device)
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        predictions = self.discriminator(generated_images)

        fake_batch_loss = self.compute_global_loss(
            labels, predictions, self.loss_weights)
        self.log("fake_batch_loss", sum(fake_batch_loss), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        print("fake batch disc train")
        #the following 3 lines correspond to
        #gradients = tape.gradient(fake_batch_loss, discriminator.trainable_variables)
        #optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) in Tensorflow
        optimizer_discriminator.zero_grad()
        self.manual_backward(sum(fake_batch_loss))
        #sum(fake_batch_loss).backward()
        optimizer_discriminator.step()

        trick = np.ones(self.batch_size).astype(np.float32)
        fake_batch = torch.tensor([[el] for el in trick]).to(self.device)
        labels = [fake_batch, energy_batch.view(-1, 1), ang_batch, ecal_batch]

        gen_losses_train = []
        # Train generator twice using combined model
        for _ in range(2):
            noise = torch.randn((self.batch_size, self.latent_size - 2)).to(self.device)
            generator_ip = torch.cat(
                (energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise),
                dim=1,)

            generated_images = self.generator(generator_ip)
            predictions = self.discriminator(generated_images)

            loss = self.compute_global_loss(
                labels, predictions, self.loss_weights)
            self.log("gen_loss", sum(loss), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            print("gen train")
            optimizer_generator.zero_grad()
            self.manual_backward(sum(loss))
            #sum(loss).backward()
            optimizer_generator.step()

            for el in loss:
                gen_losses_train.append(el)

        avg_generator_loss = sum(gen_losses_train) / len(gen_losses_train)
        self.log("generator_loss", avg_generator_loss.item(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        #avg_generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses_train)]
        #self.log("generator_loss", sum(avg_generator_loss), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        gen_losses = []
        # I'm not returning anything as in pl you do not return anything when you back-propagate manually
        #return_loss = real_batch_loss
        real_batch_loss = [real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3]]
        fake_batch_loss = [fake_batch_loss[0], fake_batch_loss[1], fake_batch_loss[2], fake_batch_loss[3]]
        gen_batch_loss = [gen_losses_train[0], gen_losses_train[1], gen_losses_train[2], gen_losses_train[3]]
        gen_losses.append(gen_batch_loss)
        gen_batch_loss = [gen_losses_train[4], gen_losses_train[5], gen_losses_train[6], gen_losses_train[7]]
        gen_losses.append(gen_batch_loss)

        real_batch_loss = [el.cpu().detach().numpy() for el in real_batch_loss]
        real_batch_loss_total_loss = np.sum(real_batch_loss)
        new_real_batch_loss = [real_batch_loss_total_loss]
        for i_weights in range(len(real_batch_loss)):
          new_real_batch_loss.append(real_batch_loss[i_weights] / self.loss_weights[i_weights])
        real_batch_loss = new_real_batch_loss

        fake_batch_loss = [el.cpu().detach().numpy() for el in fake_batch_loss]
        fake_batch_loss_total_loss = np.sum(fake_batch_loss)
        new_fake_batch_loss = [fake_batch_loss_total_loss]
        for i_weights in range(len(fake_batch_loss)):
          new_fake_batch_loss.append(fake_batch_loss[i_weights] / self.loss_weights[i_weights])
        fake_batch_loss = new_fake_batch_loss
    
        # if ecal sum has 100% loss(generating empty events) then end the training
        if fake_batch_loss[3] == 100.0 and self.index > 10:
          print("Empty image with Ecal loss equal to 100.0 for {} batch".format(self.index))
          torch.save(self.generator.state_dict(), "generator_weights.pth")
          torch.save(self.discriminator.state_dict(), "discriminator_weights.pth")
          print("real_batch_loss", real_batch_loss)
          print("fake_batch_loss", fake_batch_loss)
          sys.exit()

        # append mean of discriminator loss for real and fake events
        self.epoch_disc_loss.append([(a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)])
        
        gen_losses[0] = [el.cpu().detach().numpy() for el in gen_losses[0]]
        gen_losses_total_loss = np.sum(gen_losses[0])
        new_gen_losses = [gen_losses_total_loss]
        for i_weights in range(len(gen_losses[0])):
          new_gen_losses.append(gen_losses[0][i_weights] / self.loss_weights[i_weights])
        gen_losses[0] = new_gen_losses

        gen_losses[1] = [el.cpu().detach().numpy() for el in gen_losses[1]]
        gen_losses_total_loss = np.sum(gen_losses[1])
        new_gen_losses = [gen_losses_total_loss]
        for i_weights in range(len(gen_losses[1])):
          new_gen_losses.append(gen_losses[1][i_weights] / self.loss_weights[i_weights])
        gen_losses[1] = new_gen_losses

        generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]

        self.epoch_gen_loss.append(generator_loss)

        #self.index += 1 #this might be moved after test cycle
        
        #logging of gen and disc loss done by Trainer
        #self.log('epoch_gen_loss', self.epoch_gen_loss, on_step=True, on_epoch=True, sync_dist=True)
        #self.log('epoch_disc_loss', self.epoch_disc_loss, on_step=True, on_epoch=True, sync_dist=True)
    
    def on_train_epoch_end(self): #outputs
        epoch_num = self.current_epoch
        discriminator_train_loss = np.mean(np.array(self.epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(self.epoch_gen_loss), axis=0)

        self.train_history["generator"].append(generator_train_loss)
        self.train_history["discriminator"].append(discriminator_train_loss)

        # Log history to WandB
        wandb.log({
            "train/discriminator_loss": discriminator_train_loss[0],
            "train/generator_loss": generator_train_loss[0],
            "epoch": epoch_num
        })

        torch.save(self.generator.state_dict(), f"generator_weights_epoch_{epoch_num}.pth")
        torch.save(self.discriminator.state_dict(), f"discriminator_weights_epoch_{epoch_num}.pth")

        with open(self.pklfile, "wb") as f:
            pickle.dump({"train": self.train_history, "test": self.test_history}, f)

        #pickle.dump({"train": self.train_history}, open(self.pklfile, "wb"))
        #print("train-loss:" + str(self.train_history["generator"][-1][0]))

        print("-" * 65)
        ROW_FMT = ("{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}")
        print(ROW_FMT.format("generator (train)", *self.train_history["generator"][-1]))
        print(ROW_FMT.format("discriminator (train)", *self.train_history["discriminator"][-1]))
    
    def validation_step(self, batch, batch_idx):
        image_batch, energy_batch, ang_batch, ecal_batch = batch['X'], batch['Y'], batch['ang'], batch['ecal']

        image_batch = image_batch.permute(0, 4, 1, 2, 3)

        image_batch = image_batch.to(self.device)
        energy_batch = energy_batch.to(self.device)
        ang_batch = ang_batch.to(self.device)
        ecal_batch = ecal_batch.to(self.device)

        # Generate Fake events with same energy and angle as data batch        
        noise = torch.randn((self.batch_size, self.latent_size - 2), dtype=torch.float32).to(self.device)
        
        generator_ip = torch.cat((energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise), dim=1)
        generated_images = self.generator(generator_ip)

        # concatenate to fake and real batches
        X = torch.cat((image_batch, generated_images), dim=0)

        #y = np.array([1] * self.batch_size + [0] * self.batch_size).astype(np.float32)
        y = torch.tensor([1] * self.batch_size + [0] * self.batch_size, dtype=torch.float32).to(self.device)
        y = y.view(-1, 1)

        ang = torch.cat((ang_batch, ang_batch), dim=0)
        ecal = torch.cat((ecal_batch, ecal_batch), dim=0)
        aux_y = torch.cat((energy_batch, energy_batch), dim=0)

        #y = [[el] for el in y]
        labels = [y, aux_y, ang, ecal]

        # Calculate discriminator loss
        disc_eval = self.discriminator(X)
        disc_eval_loss = self.compute_global_loss(labels, disc_eval, self.loss_weights)
        
        # Calculate generator loss
        trick = np.ones(self.batch_size).astype(np.float32)
        fake_batch = torch.tensor([[el] for el in trick]).to(self.device)
        #fake_batch = [[el] for el in trick]
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        generated_images = self.generator(generator_ip)
        gen_eval = self.discriminator(generated_images)
        gen_eval_loss = self.compute_global_loss(labels, gen_eval, self.loss_weights)

        self.log('val_discriminator_loss', sum(disc_eval_loss), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_generator_loss', sum(gen_eval_loss), on_epoch=True, prog_bar=True, sync_dist=True)

        disc_test_loss = [disc_eval_loss[0], disc_eval_loss[1], disc_eval_loss[2], disc_eval_loss[3]]
        gen_test_loss = [gen_eval_loss[0], gen_eval_loss[1], gen_eval_loss[2], gen_eval_loss[3]]

        # Configure the loss so it is equal to the original values
        disc_eval_loss = [el.cpu().detach().numpy() for el in disc_test_loss]
        disc_eval_loss_total_loss = np.sum(disc_eval_loss)
        new_disc_eval_loss = [disc_eval_loss_total_loss]
        for i_weights in range(len(disc_eval_loss)):
            new_disc_eval_loss.append(disc_eval_loss[i_weights] / self.loss_weights[i_weights])
        disc_eval_loss = new_disc_eval_loss

        gen_eval_loss = [el.cpu().detach().numpy() for el in gen_test_loss]
        gen_eval_loss_total_loss = np.sum(gen_eval_loss)
        new_gen_eval_loss = [gen_eval_loss_total_loss]
        for i_weights in range(len(gen_eval_loss)):
            new_gen_eval_loss.append(gen_eval_loss[i_weights] / self.loss_weights[i_weights])
        gen_eval_loss = new_gen_eval_loss

        self.index += 1
        # evaluate discriminator loss
        self.disc_epoch_test_loss.append(disc_eval_loss)
        # evaluate generator loss
        self.gen_epoch_test_loss.append(gen_eval_loss)

    def on_validation_epoch_end(self):
        discriminator_test_loss = np.mean(np.array(self.disc_epoch_test_loss), axis=0)
        generator_test_loss = np.mean(np.array(self.gen_epoch_test_loss), axis=0)

        self.test_history["generator"].append(generator_test_loss)
        self.test_history["discriminator"].append(discriminator_test_loss)

        wandb.log({
        "val/discriminator_loss": discriminator_test_loss[0],
        "val/generator_loss": generator_test_loss[0],
        "epoch": self.current_epoch
    })

        print("-" * 65)
        ROW_FMT = ("{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}")
        print(ROW_FMT.format("generator (test)", *self.test_history["generator"][-1]))
        print(ROW_FMT.format("discriminator (test)", *self.test_history["discriminator"][-1]))

        # save loss dict to pkl file
        with open(self.pklfile, "wb") as f:
            pickle.dump({"train": self.train_history, "test": self.test_history}, f)
        #pickle.dump({"test": self.test_history}, open(self.pklfile, "wb"))
        #print("train-loss:" + str(self.train_history["generator"][-1][0]))

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> Any:
        energy_batch, ang_batch = batch['Y'], batch['ang']

        energy_batch = energy_batch.to(self.device)
        ang_batch = ang_batch.to(self.device)

        # Generate Fake events with same energy and angle as data batch
        noise = torch.randn(
            (energy_batch.shape[0], self.latent_size - 2),
            dtype=torch.float32,
            device=self.device
        )

        # print(f"Reshape energy: {energy_batch.view(-1, 1).shape}")
        # print(f"Reshape angle: {ang_batch.view(-1, 1).shape}")
        # print(f"Noise: {noise.shape}")

        generator_ip = torch.cat(
            [energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise],
            dim=1
        )
        # print(f"Generator input: {generator_ip.shape}")
        generated_images = self.generator(generator_ip)
        # print(f"Generated batch size {generated_images.shape}")
        return {'images': generated_images,
                'energies': energy_batch,
                'angles': ang_batch}
    
    def configure_optimizers(self):
        lr = self.hparams.lr

        optimizer_discriminator = torch.optim.RMSprop(self.discriminator.parameters(), lr, alpha=0.9, eps=1e-07)
        optimizer_generator = torch.optim.RMSprop(self.generator.parameters(), lr, alpha=0.9, eps=1e-07)
        return [optimizer_discriminator, optimizer_generator], []


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

wandb_logger = WandbLogger(project='3dgan_training')

data = ParticlesDataModule()
model = ThreeDGAN()
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=30,
    logger=wandb_logger
)
trainer.fit(model, data)

wandb_logger.log_hyperparams(model.hparams)

'''
trained_generator = model.generator
trained_generator.eval()

print('analysing..........')
energies = [0, 110, 150, 190]
analysis_history = defaultdict(list)
resultfile = "/eos/user/k/ktsolaki/misc/3dgan_pytorch/3dgan_analysis_test1.pkl"
analysis_loader = data.val_dataloader()
nb_Test = len(analysis_loader.dataset)
atime = time.time()

# load all test data
# for index, dtest in enumerate(analysis_loader):
#     if index == 0:
#         print("Keys in the batch:", dtest.keys())
#         X_test, Y_test, ang_test, ecal_test = GetDataAngle(dtest, xscale=1, angscale=1, angtype="mtheta", thresh=0, daxis=1)
#     else:
#         if X_test.shape[0] < nb_Test:
#             X_temp, Y_temp, ang_temp,  ecal_temp = GetDataAngle(dtest, xscale=1, angscale=1, angtype="mtheta", thresh=0, daxis=1)
#             X_test = np.concatenate((X_test, X_temp))
#             Y_test = np.concatenate((Y_test, Y_temp))
#             ang_test = np.concatenate((ang_test, ang_temp))
#             ecal_test = np.concatenate((ecal_test, ecal_temp))
# if X_test.shape[0] > nb_Test:
#     X_test, Y_test, ang_test, ecal_test = X_test[:nb_Test], Y_test[:nb_Test], ang_test[:nb_Test], ecal_test[:nb_Test]
# else:
#     nb_Test = X_test.shape[0] # the nb_test maybe different if total events are less than nEvents
# var = sortEnergy([np.squeeze(X_test), Y_test, ang_test], ecal_test, energies, ang=1)
# print(var.keys())
# result = OptAnalysisAngle(var, trained_generator, energies, xpower = 0.85, concat=2)
# print('{} seconds taken by analysis'.format(time.time()-atime))
# analysis_history['total'].append(result[0])
# analysis_history['energy'].append(result[1])
# analysis_history['moment'].append(result[2])
# analysis_history['angle'].append(result[3])
# print('Result = ', result)
# # write analysis history to a pickel file
# pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

X_test = []
Y_test = []
ang_test = []
ecal_test = []

for batch in analysis_loader:
    X_test.append(batch['X'])
    Y_test.append(batch['Y'])
    ang_test.append(batch['ang'])
    ecal_test.append(batch['ecal'])

X_test = torch.cat(X_test, dim=0)
Y_test = torch.cat(Y_test, dim=0)
ang_test = torch.cat(ang_test, dim=0)
ecal_test = torch.cat(ecal_test, dim=0)

# min_value = 0
# max_value = 1
# # Check if elements are within the range
# within_range_mask = (Y_test >= min_value) & (Y_test <= max_value)
# # Check if any element is within the range
# has_values_within_range = within_range_mask.any()
# print(f"Contains values within range {min_value} to {max_value}: {has_values_within_range.item()}")


# print(X_test.shape)
# print(Y_test.shape)
# print(ang_test.shape)
# print(ecal_test.shape)

var = sortEnergy([np.squeeze(X_test), Y_test, ang_test], ecal_test, energies, ang=1)
#print(var.keys())
# print(var["events_act0"].shape)
# print(var["events_act110"].shape)
# print(var["events_act150"].shape)
# print(var["events_act190"].shape)

# print(var["index0"])
# print(var["indexes110"])
# print(var["indexes150"])
# print(var["indexes190"])

result = OptAnalysisAngle(var, trained_generator, energies, xpower = 0.85, concat=2)
print('{} seconds taken by analysis'.format(time.time()-atime))

analysis_history['total'].append(result[0])
analysis_history['energy'].append(result[1])
analysis_history['moment'].append(result[2])
analysis_history['angle'].append(result[3])
print('Result = ', result)

# write analysis history to a pickle file
pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))
'''
