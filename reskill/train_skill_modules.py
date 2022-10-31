
import torch
import torch.optim as optim
import argparse
from typing import List
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import pdb
import wandb
from tqdm import tqdm
import os
import time
import yaml

from reskill.models.skill_vae import SkillVAE
from reskill.data.skill_dataloader import SkillsDataset
from reskill.models.rnvp import stacked_NVP
from reskill.utils.general_utils import AttrDict



class ModelTrainer():
    def __init__(self, dataset_name, config_file):
        self.dataset_name = dataset_name
        self.save_dir = "./results/saved_skill_models/" + dataset_name +"/"
        os.makedirs(self.save_dir, exist_ok=True)
        self.vae_save_path = self.save_dir + "skill_vae.pth"
        self.sp_save_path = self.save_dir + "skill_prior.pth"
        config_path = "configs/skill_mdl/" + config_file

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)


        with open(config_path, 'r') as file:
            conf = yaml.safe_load(file)
            conf = AttrDict(conf)
        for key in conf:
            conf[key] = AttrDict(conf[key])        

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])          
        train_data = SkillsDataset(dataset_name, phase="train", subseq_len=conf.skill_vae.subseq_len, transform=transform)
        val_data   = SkillsDataset(dataset_name, phase="val", subseq_len=conf.skill_vae.subseq_len, transform=transform)

        self.train_loader = DataLoader(
            train_data,
            batch_size = conf.skill_vae.batch_size,
            shuffle = True,
            drop_last=True,
            prefetch_factor=30,
            num_workers=8,
            pin_memory=True)

        self.val_loader = DataLoader(
            val_data,
            batch_size = 64,
            shuffle = False,
            drop_last=True,
            prefetch_factor=30,
            num_workers=8,
            pin_memory=True)

        self.skill_vae = SkillVAE(n_actions=conf.skill_vae.n_actions, n_obs=conf.skill_vae.n_obs, n_hidden=conf.skill_vae.n_hidden,
                                  seq_length=conf.skill_vae.subseq_len, n_z=conf.skill_vae.n_z, device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.skill_vae.parameters(), lr=conf.skill_vae.lr)

        self.sp_nvp = stacked_NVP(d=conf.skill_prior_nvp.d, k=conf.skill_prior_nvp.k, n_hidden=conf.skill_prior_nvp.n_hidden,
                                  state_size=conf.skill_vae.n_obs, n=conf.skill_prior_nvp.n_coupling_layers, device=self.device).to(self.device)
        
        self.sp_optimizer = torch.optim.Adam(self.sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.sp_optimizer, 0.999)

        self.n_epochs = conf.skill_vae.epochs


    def fit(self, epoch):
        self.skill_vae.train()
        running_loss = 0.0
        for i, data in enumerate(self.train_loader):

            data["actions"] = data["actions"].to(self.device)
            data["obs"] = data["obs"].to(self.device)

            # Train skills model
            self.skill_vae.init_hidden(data["actions"].size(0))
            self.optimizer.zero_grad()
            output = self.skill_vae(data)
            losses = self.skill_vae.loss(data, output)
            loss = losses.total_loss
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # Train skills prior model
            self.sp_optimizer.zero_grad()
            sp_input = AttrDict(skill=output.z.detach(),
                                state=data["obs"][:,0,:])
            z, log_pz, log_jacob = self.sp_nvp(sp_input)
            sp_loss = (-log_pz - log_jacob).mean()
            sp_loss.backward()
            self.sp_optimizer.step()

            if i%500 == 0:
                self.scheduler.step()
                wandb.log({'lr':self.scheduler.get_lr()[0]}, epoch)

            if i % 100:
                wandb.log({'BC Loss_VAE':losses.bc_loss.item()}, epoch)
                wandb.log({'KL_Loss_VAE':losses.kld_loss.item()}, epoch)
                wandb.log({'NVP_Loss':sp_loss.item()}, epoch)
            
        train_loss = running_loss/len(self.train_loader.dataset)
        return train_loss


    def validate(self):
        self.skill_vae.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                data["actions"] = data["actions"].to(self.device)
                data["obs"] = data["obs"].to(self.device)
                self.skill_vae.init_hidden(data["actions"].size(0))
                self.optimizer.zero_grad()
                output = self.skill_vae(data)
                losses = self.skill_vae.loss(data, output)

                loss = losses.bc_loss.item()
                running_loss += loss

        val_loss = running_loss/len(self.val_loader.dataset)
        return val_loss


    def train(self):
        print("Training...") 
        for epoch in tqdm(range(self.n_epochs)):
            train_epoch_loss = self.fit(epoch)
            if epoch%5 == 0:
                val_epoch_loss = self.validate()

            wandb.log({'train_loss':train_epoch_loss}, epoch)
            wandb.log({'val_loss':val_epoch_loss}, epoch)

            if epoch % 50 == 0:
                torch.save(self.skill_vae, self.vae_save_path)
                torch.save(self.sp_nvp, self.sp_save_path)
                
   
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="block/config.yaml")
    parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")
    args=parser.parse_args()
    
    wandb.init(project="skill_mdl")
    wandb.run.name = "skill_mdl_" + time.asctime()
    wandb.run.save()

    trainer = ModelTrainer(args.dataset_name, args.config_file)
    trainer.train()