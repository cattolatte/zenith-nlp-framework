import os
import torch
from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from ..evaluation.metrics import accuracy
import mlflow
import mlflow.pytorch

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_model(model, train_dataset, val_dataset, epochs, learning_rate, batch_size,
                optimizer_name='adamw', warmup_steps=500, grad_clip_value=1.0, save_path="model.pth", task_type='classification',
                model_params=None): # Added model_params to log them
    """
    A comprehensive training loop with MLflow integration.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_distributed:
        setup_distributed(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and is_distributed else "cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0: # Only log from the main process
        mlflow.start_run()
        
        # Log hyperparameters
        params_to_log = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer_name,
            "task_type": task_type,
        }
        if model_params:
            params_to_log.update(model_params)
        mlflow.log_params(params_to_log)

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    
    optimizer = AdamW(model.parameters(), lr=learning_rate) if optimizer_name.lower() == 'adamw' else Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss(ignore_index=-100)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    for epoch in range(epochs):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # (Task-specific loss calculation remains the same)
            if task_type in ['classification', 'ner']:
                inputs = {'ids': batch['ids'], 'segment_info': batch.get('segment_info')}
                labels = batch.get('labels') or batch.get('label')
                outputs = model(**inputs)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            elif task_type == 'question_answering':
                inputs = {'ids': batch['ids'], 'segment_info': batch.get('segment_info')}
                start_positions, end_positions = batch['start_positions'], batch['end_positions']
                start_logits, end_logits = model(**inputs)
                loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2
            elif task_type == 'summarization':
                inputs = {'src': batch['src'], 'trg': batch['trg']}
                labels = batch['label']
                outputs = model(**inputs)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        # (Validation loop remains the same)
        model.eval()
        total_val_loss, total_val_acc = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                if task_type in ['classification', 'ner']:
                    inputs = {'ids': batch['ids'], 'segment_info': batch.get('segment_info')}
                    labels = batch.get('labels') or batch.get('label')
                    outputs = model(**inputs)
                    loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                    if task_type == 'classification':
                        total_val_acc += accuracy(outputs, labels)
                else: # Placeholder loss for other tasks
                    loss = torch.tensor(0.0)
                total_val_loss += loss.item()

        if rank == 0:
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            avg_val_acc = (total_val_acc / len(val_loader)) * 100 if len(val_loader) > 0 else 0
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", avg_val_acc, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")

    if rank == 0:
        model_to_save = model.module if is_distributed else model
        torch.save(model_to_save.state_dict(), save_path)
        # Log the trained model as an artifact
        mlflow.pytorch.log_model(model_to_save, "model")
        print("Model saved and logged to MLflow.")
        mlflow.end_run()

    if is_distributed:
        cleanup_distributed()
