from tqdm import tqdm 
import torch
from src.utils import  calculate_iou, calculate_dice_coefficient
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, 
                 criterion, optimizer, device,tb_path, checkpoint_path):
        """
        Training Process
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.tb_path = tb_path
        self.best_valid_loss = float('inf')
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
        }
        checkpoint_filename = f'models/{self.checkpoint_path}/checkpoint_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_filename)
        print(f"Checkpoint saved at epoch {epoch+1}")
        
        if is_best:
            best_model_filename = 'best_model.pt'
            torch.save(checkpoint, best_model_filename)
            print(f"Best model saved at epoch {epoch+1}")

    def train(self, epochs):
        writer = SummaryWriter(self.tb_path)
        for epoch in range(epochs):
            self.model.train()
            loop = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            total_loss = 0.0
            total_iou = 0.0
            total_dice = 0.0
            for batch_idx, (data, targets) in enumerate(loop):
                data, targets = data.to(self.device), targets.to(self.device)
                # foward
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                dice_coefficient = calculate_dice_coefficient(torch.sigmoid(outputs) > 0.5, 
                                                                  targets > 0.5)
                # backward
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_dice += dice_coefficient.item()
            avg_loss = total_loss / len(self.train_dataloader)
            avg_tdice = total_dice / len(self.train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Training- Loss: {avg_loss:.4f}, Dice: {avg_tdice:.4f}")
            
            #writer.add_scalar('Train Loss', avg_loss, epoch)
            # Save checkpoint after each epoch
            self.save_checkpoint(epoch) #is_best=True
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                total_iou = 0.0
                total_dice = 0.0
                total_vloss = 0.0
                for data, targets in self.valid_dataloader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    #Loss matrix
                    val_loss = self.criterion(outputs, targets)
                    iou = calculate_iou(torch.sigmoid(outputs) > 0.5, targets > 0.5)
                    dice_coefficient = calculate_dice_coefficient(torch.sigmoid(outputs) > 0.5, 
                                                                  targets > 0.5)
                    #Total Loss & cofficient 
                    total_iou += iou.item()
                    total_dice += dice_coefficient.item()
                    total_vloss += val_loss.item()
                avg_vloss = total_vloss / len(self.valid_dataloader)
                avg_iou = total_iou / len(self.valid_dataloader)
                avg_vdice = total_dice / len(self.valid_dataloader)
                # Log the running loss averaged per batch
                writer.add_scalars('Training vs. Validation Loss',
                                { 'Training' : avg_loss, 'Validation' : avg_vloss },
                                epoch)
                writer.add_scalars('Training vs. Validation Dice',
                                { 'Training' : avg_tdice, 'Validation' : avg_vdice },
                                epoch)
                #writer.add_scalar('Valid Loss', val_avg_loss, epoch)
                #writer.add_scalar('Valid Dice', avg_dice, epoch)
                #self.valid_losses.append(avg_loss)  # Store the validation loss
                print(f"Validation- Loss: {avg_vloss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_vdice:.4f}")

                # Check if the current model has the best validation loss
                if avg_loss < self.best_valid_loss:
                    self.best_valid_loss = avg_loss
                    self.save_checkpoint(epoch, is_best=True)
        print('Finished Training')
        writer.flush()