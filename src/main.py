from typing import Union
from types import SimpleNamespace
import argparse
import os
import sys
from sklearn.metrics import roc_auc_score, accuracy_score

import numpy as np
import torch
import torchvision
from tinyimagenet import TinyImageNet
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/model/')
from timm_models import build_timm_model
sys.path.insert(0, import_dir + '/src/nn_utils/')
from log import log
from seed import seed_everything
from ssl_aug import SingleInstanceTwoView
from scheduler import LinearWarmupCosineAnnealingLR
from extend import ExtendedDataset


def count_parameters(model, trainable_only: bool = False):
    if not trainable_only:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataloaders(args: SimpleNamespace):
    if args.dataset == 'mnist':
        args.in_channels = 1
        args.num_classes = 10
        dataset_mean = (0.1307, )
        dataset_std = (0.3081, )
        torchvision_dataset = torchvision.datasets.MNIST

    elif args.dataset == 'cifar10':
        args.in_channels = 3
        args.num_classes = 10
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR10

    elif args.dataset == 'stl10':
        args.in_channels = 3
        args.num_classes = 10
        dataset_mean = (0.4467, 0.4398, 0.4066)
        dataset_std = (0.2603, 0.2566, 0.2713)
        torchvision_dataset = torchvision.datasets.STL10

    elif args.dataset == 'tinyimagenet':
        args.in_channels = 3
        args.num_classes = 200
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = TinyImageNet

    elif args.dataset == 'imagenet':
        args.in_channels = 3
        args.num_classes = 1000
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = torchvision.datasets.ImageNet

    else:
        raise ValueError(
            '`config.dataset` value not supported. Value provided: %s.' %
            args.dataset)

    # NOTE: To accommodate the ViT models, we resize all images to 224x224.
    imsize = 224

    if args.in_channels == 3:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                imsize,
                interpolation=torchvision.transforms.InterpolationMode.
                BICUBIC),
            torchvision.transforms.RandomResizedCrop(
                imsize,
                scale=(0.6, 1.6),
                interpolation=torchvision.transforms.InterpolationMode.
                BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ],
                                                p=0.4),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=dataset_mean,
                                                std=dataset_std)
        ])
    else:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                imsize,
                interpolation=torchvision.transforms.InterpolationMode.
                BICUBIC),
            torchvision.transforms.RandomResizedCrop(
                imsize,
                scale=(0.6, 1.6),
                interpolation=torchvision.transforms.InterpolationMode.
                BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=dataset_mean,
                                                std=dataset_std)
        ])

    transform_train_ssl = SingleInstanceTwoView(imsize=imsize,
                                                mean=dataset_mean,
                                                std=dataset_std)

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            imsize,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(imsize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    if args.dataset in ['mnist', 'cifar10', 'cifar100']:
        train_set = torchvision_dataset(args.dataset_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_train)
        train_set_ssl = torchvision_dataset(args.dataset_dir,
                                            train=True,
                                            download=True,
                                            transform=transform_train_ssl)
        test_set = torchvision_dataset(args.dataset_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test)

    elif args.dataset in ['stanfordcars', 'stl10', 'food101', 'flowers102']:
        train_set = torchvision_dataset(args.dataset_dir,
                                        split='train',
                                        download=True,
                                        transform=transform_train)
        train_set_ssl = torchvision_dataset(args.dataset_dir,
                                            split='train',
                                            download=True,
                                            transform=transform_train_ssl)
        test_set = torchvision_dataset(args.dataset_dir,
                                       split='test',
                                       download=True,
                                       transform=transform_test)

        if args.dataset == 'stl10':
            # Training set has too few images (5000 images in total).
            # Let's augment it into a bigger dataset.
            train_set = ExtendedDataset(train_set,
                                        desired_len=10 *
                                        len(train_set))
            train_set_ssl = ExtendedDataset(train_set,
                                            desired_len=10 *
                                            len(train_set_ssl))

    elif args.dataset in ['tinyimagenet', 'imagenet']:
        train_set = torchvision_dataset(args.dataset_dir,
                                        split='train',
                                        transform=transform_train)
        train_set_ssl = torchvision_dataset(args.dataset_dir,
                                            split='train',
                                            transform=transform_train_ssl)
        test_set = torchvision_dataset(args.dataset_dir,
                                       split='val',
                                       transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)
    train_loader_ssl = torch.utils.data.DataLoader(train_set_ssl,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)

    return (train_loader, train_loader_ssl, val_loader), args


def simsiam(p1_batch: torch.Tensor,
            p2_batch: torch.Tensor,
            z1_batch: torch.Tensor,
            z2_batch: torch.Tensor,
            **kwargs) -> torch.Tensor:
    '''
    SimSiam loss.
    (Algorithm 1 in the SimSiam paper https://arxiv.org/abs/2011.10566).
    '''
    loss = __neg_cos_sim(p1_batch, z2_batch) / 2 + __neg_cos_sim(p2_batch, z1_batch) / 2
    return loss

def __neg_cos_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    '''
    Negative cosine similarity as in SimSiam.
    '''
    z = z.detach() # stop gradient
    p = torch.nn.functional.normalize(p, p=2, dim=1)  # L2-normalize
    z = torch.nn.functional.normalize(z, p=2, dim=1)  # L2-normalize
    return -(p * z).sum(dim=1).mean()

def simclr(p1_batch: torch.Tensor,
           p2_batch: torch.Tensor,
           temperature: float = 0.5,
           **kwargs) -> torch.Tensor:
    '''
    SimCLR loss.
    (Equation 1 in the SimCLR paper https://arxiv.org/abs/2002.05709).
    '''
    assert p1_batch.shape == p2_batch.shape
    B, _ = p1_batch.shape

    p1_batch = torch.nn.functional.normalize(p1_batch, p=2, dim=1)
    p2_batch = torch.nn.functional.normalize(p2_batch, p=2, dim=1)
    p = torch.cat((p1_batch, p2_batch), dim=0)  # 2B x D

    # Compute similarity matrix
    # Note that we refactor the `exp` and `1/temperature` operations here.
    sim_matrix = torch.exp(torch.matmul(p, p.T) / temperature)

    # Masks to identify positive and negative examples.
    positive_mask = torch.cat((
        torch.cat((torch.zeros((B, B), dtype=torch.bool), torch.eye(B, dtype=torch.bool)), dim=0),
        torch.cat((torch.eye(B, dtype=torch.bool), torch.zeros((B, B), dtype=torch.bool)), dim=0),
                            ), dim=1)
    negative_mask = torch.cat((
        torch.cat((~torch.eye(B, dtype=torch.bool), ~torch.eye(B, dtype=torch.bool)), dim=0),
        torch.cat((~torch.eye(B, dtype=torch.bool), ~torch.eye(B, dtype=torch.bool)), dim=0),
                            ), dim=1)

    # Select the positive examples.
    score_pos = sim_matrix[positive_mask].view(2 * B, 1)

    # Sum all similarities for negative pairs.
    score_neg = sim_matrix[negative_mask].view(2 * B, -1).sum(dim=1, keepdim=True)

    # Calculate the InfoNCE loss as the log ratio.
    loss = -torch.log(score_pos / (score_pos + score_neg))
    loss = loss.mean() / B
    return loss

def barlow_twins(p1_batch: torch.Tensor,
                 p2_batch: torch.Tensor,
                 lambda_off_diag: float = 1e-2,
                 **kwargs) -> torch.Tensor:
    '''
    Barlow twins loss.
    (Algorithm 1 in the Barlow Twins paper https://arxiv.org/pdf/2103.03230).

    There are two components, (1) pairwise similarity and (2) feature non-redundancy.
    '''
    assert p1_batch.shape == p2_batch.shape
    B, D = p1_batch.shape

    # Mean centering.
    p1_batch = p1_batch - p1_batch.mean(0)                   # B x D
    p2_batch = p2_batch - p2_batch.mean(0)                   # B x D

    # Unit variance regularization.
    std_p1 = torch.sqrt(p1_batch.var(dim=0) + 0.0001)
    std_p2 = torch.sqrt(p2_batch.var(dim=0) + 0.0001)
    std_loss = (torch.mean(torch.nn.functional.relu(1 - std_p1)) + torch.mean(torch.nn.functional.relu(1 - std_p2))) / 2

    # Pair similarity matrix.
    pair_sim = torch.mm(p1_batch, p2_batch.T) / D            # B x B

    # Cross-correlation matrix.
    cross_corr = torch.mm(p1_batch.T, p2_batch) / B          # D x D

    # Push and pull based on immunogenicity.
    pair_sim_ideal = torch.eye(B, device=p1_batch.device)    # B x B
    pair_sim_diff = (pair_sim - pair_sim_ideal).pow(2)       # B x B

    # Down-weigh the off-diagonal items.
    pair_sim_diff[~torch.eye(B, dtype=bool)] *= lambda_off_diag

    # Encourage pair correlation and reduce feature redundancy.
    cross_corr_ideal = torch.eye(D, device=p1_batch.device)   # D x D
    cross_corr_diff = (cross_corr - cross_corr_ideal).pow(2)  # D x D
    # Down-weigh the off-diagonal items.
    cross_corr_diff[~torch.eye(D, dtype=bool)] *= lambda_off_diag

    loss = pair_sim_diff.sum() / B + cross_corr_diff.sum() / D + std_loss
    return loss


def main(args: SimpleNamespace) -> None:
    '''
    The main function of training and evaluation.
    1. Train for `args.epochs_pretrain` epochs. No validation set.
    2. Linear probe or fine-tune for `args.epochs_tuning` epochs.
    3. Evaluate on test set.
    '''

    # Log the config.
    config_str = 'Config: \n'
    args_dict = args.__dict__
    for key in args_dict.keys():
        config_str += '%s: %s\n' % (key, args_dict[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=args.log_path, to_console=False)

    seed_everything(args.random_seed)

    dataloaders, args = get_dataloaders(args=args)
    train_loader, train_loader_ssl, test_loader = dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_timm_model(model_name=args.model,
                             num_classes=args.num_classes).to(device)
    model.init_params()
    model.to(device)

    infer(model=model, loader=test_loader, loss_fn_pred=torch.nn.CrossEntropyLoss(), device=device)

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_pretrain_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_linear_probe_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_finetune_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.npz_save_path), exist_ok=True)

    # Loss function.
    loss_fn_pred = torch.nn.CrossEntropyLoss()
    if args.learning_method == 'supervised':
        loss_fn = loss_fn_pred
    elif args.learning_method == 'simclr':
        loss_fn = simclr
    elif args.learning_method == 'simsiam':
        loss_fn = simsiam
    elif args.learning_method == 'barlow_twins':
        loss_fn = barlow_twins
    else:
        raise ValueError(f'loss function `{args.learning_method}` not supported.')

    if os.path.isfile(args.model_pretrain_save_path):
        log(f'Step 1 skipped. Pre-trained model found at {args.model_pretrain_save_path}.',
            filepath=args.log_path, to_console=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr_pretrain))
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                                    warmup_epochs=min(20, args.epochs_pretrain // 2),
                                                    warmup_start_lr=args.lr_pretrain * 1e-2,
                                                    max_epochs=args.epochs_pretrain)

        # Step 1. Train for `args.epochs_pretrain` epochs. No validation set.
        train_loss_arr, train_acc_arr, train_auroc_arr = [], [], []
        log('Step 1. Pre-training.', filepath=args.log_path, to_console=True)
        for epoch_idx in tqdm(range(args.epochs_pretrain)):
            if args.learning_method == 'supervised':
                loader = train_loader
            else:
                loader = train_loader_ssl
            model, optimizer, lr_scheduler, loss, acc, auroc = \
                train_epoch(model=model, loader=loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
                            learning_method=args.learning_method, loss_fn=loss_fn, device=device)
            log(f'[Epoch {epoch_idx+1}/{args.epochs_pretrain}]. LR={optimizer.param_groups[0]['lr']},' + \
                f'Training loss={loss:.4f}, ACC={acc:.3f}, AUROC={auroc:.3f}.', filepath=args.log_path, to_console=True)
            train_loss_arr.append(loss)
            train_acc_arr.append(acc)
            train_auroc_arr.append(auroc)
        torch.save(model.state_dict(), args.model_pretrain_save_path)

    # Step 2. Linear probe or fine-tune for `args.epochs_tuning` epochs.
    if os.path.isfile(args.model_linear_probe_save_path) and os.path.isfile(args.model_finetune_save_path):
        log(f'Step 2 skipped. Linear probed model found at {args.model_linear_probe_save_path} and fine-tuned model found at {args.model_finetune_save_path}.',
            filepath=args.log_path, to_console=True)
    else:
        linear_probe_loss_arr, linear_probe_acc_arr, linear_probe_auroc_arr = [], [], []
        finetune_loss_arr, finetune_acc_arr, finetune_auroc_arr = [], [], []
        for tuning_method in ['linear_probe', 'finetune']:
            if args.learning_method == 'supervised':
                log('Step 2 skipped. Supervised learning does not need this step.', filepath=args.log_path, to_console=True)
            else:
                model.load_state_dict(torch.load(args.model_pretrain_save_path, weights_only=True))
                log(f'Loaded pre-trained model from {args.model_pretrain_save_path}.',
                    filepath=args.log_path, to_console=True)
                if tuning_method == 'linear_probe':
                    log('Step 2. Linear Probing.', filepath=args.log_path, to_console=True)
                    # Linear probing. Only update the last linear layer.
                    model.freeze_encoder()
                    optimizer = torch.optim.AdamW(model.linear.parameters(), lr=float(args.lr_tuning))
                else:
                    log('Step 2. Fine-tuning.', filepath=args.log_path, to_console=True)
                    # Fine-tuning. Updates the entire model.
                    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr_tuning))

                lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                                            warmup_epochs=min(20, args.epochs_tuning // 2),
                                                            warmup_start_lr=args.lr_tuning * 1e-2,
                                                            max_epochs=args.epochs_tuning)
                for epoch_idx in tqdm(range(args.epochs_tuning)):
                    model, optimizer, lr_scheduler, loss, acc, auroc = \
                        train_epoch(model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                    learning_method='supervised', loss_fn=loss_fn_pred, device=device)
                    log(f'[Epoch {epoch_idx+1}/{args.epochs_tuning}]. LR={optimizer.param_groups[0]['lr']},' + \
                        f'Tuning loss={loss:.4f}, ACC={acc:.3f}, AUROC={auroc:.3f}.', filepath=args.log_path, to_console=True)
                    if tuning_method == 'linear_probe':
                        linear_probe_loss_arr.append(loss)
                        linear_probe_acc_arr.append(acc)
                        linear_probe_auroc_arr.append(auroc)
                    else:
                        finetune_loss_arr.append(loss)
                        finetune_acc_arr.append(acc)
                        finetune_auroc_arr.append(auroc)
                if tuning_method == 'linear_probe':
                    torch.save(model.state_dict(), args.model_linear_probe_save_path)
                else:
                    torch.save(model.state_dict(), args.model_finetune_save_path)

    # Step 3. Evaluate on test set.
    if args.learning_method == 'supervised':
        model.load_state_dict(torch.load(args.model_pretrain_save_path, weights_only=True))
        log(f'Supervised training. Loaded pre-trained model from {args.model_pretrain_save_path}.',
            filepath=args.log_path, to_console=True)
        linear_probe_eval_loss, linear_probe_eval_acc, linear_probe_eval_auroc = -1, -1, -1
        finetune_eval_loss, finetune_eval_acc, finetune_eval_auroc = -1, -1, -1
        supervised_eval_loss, supervised_eval_acc, supervised_eval_auroc = \
            infer(model=model, loader=test_loader, loss_fn_pred=loss_fn_pred, device=device)
        log(f'[Supervised Evaluation] loss={supervised_eval_loss:.4f}, ACC={supervised_eval_acc:.3f}, AUROC={supervised_eval_auroc:.3f}.',
            filepath=args.log_path, to_console=True)

    else:
        model.load_state_dict(torch.load(args.model_linear_probe_save_path, weights_only=True))
        log(f'Loaded linear probed model from {args.model_linear_probe_save_path}.',
            filepath=args.log_path, to_console=True)
        linear_probe_eval_loss, linear_probe_eval_acc, linear_probe_eval_auroc = \
            infer(model=model, loader=test_loader, loss_fn_pred=loss_fn_pred, device=device)
        log(f'[Linear Probing Evaluation] loss={linear_probe_eval_loss:.4f}, ACC={linear_probe_eval_acc:.3f}, AUROC={linear_probe_eval_auroc:.3f}.',
            filepath=args.log_path, to_console=True)

        model.load_state_dict(torch.load(args.model_finetune_save_path, weights_only=True))
        log(f'Loaded fine-tuned model from {args.model_finetune_save_path}.',
            filepath=args.log_path, to_console=True)
        finetune_eval_loss, finetune_eval_acc, finetune_eval_auroc = \
            infer(model=model, loader=test_loader, loss_fn_pred=loss_fn_pred, device=device)
        log(f'[Fine-tuning Evaluation] loss={linear_probe_eval_loss:.4f}, ACC={linear_probe_eval_acc:.3f}, AUROC={linear_probe_eval_auroc:.3f}.',
            filepath=args.log_path, to_console=True)

    # Save the results after training.
    with open(args.npz_save_path, 'wb+') as f:
        np.savez(
            f,
            train_loss_arr=np.array(train_loss_arr),
            train_acc_arr=np.array(train_acc_arr),
            train_auroc_arr=np.array(train_auroc_arr),
            linear_probe_loss_arr=np.array(linear_probe_loss_arr),
            linear_probe_acc_arr=np.array(linear_probe_acc_arr),
            linear_probe_auroc_arr=np.array(linear_probe_auroc_arr),
            finetune_loss_arr=np.array(finetune_loss_arr),
            finetune_acc_arr=np.array(finetune_acc_arr),
            finetune_auroc_arr=np.array(finetune_auroc_arr),
            supervised_eval_loss=np.array(supervised_eval_loss),
            supervised_eval_acc=np.array(supervised_eval_acc),
            supervised_eval_auroc=np.array(supervised_eval_auroc),
            linear_probe_eval_loss=np.array(linear_probe_eval_loss),
            linear_probe_eval_acc=np.array(linear_probe_eval_acc),
            linear_probe_eval_auroc=np.array(linear_probe_eval_auroc),
            finetune_eval_loss=np.array(finetune_eval_loss),
            finetune_eval_acc=np.array(finetune_eval_acc),
            finetune_eval_auroc=np.array(finetune_eval_auroc),
        )

    return


def train_epoch(model: torch.nn.Module,
                loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                learning_method: str,
                loss_fn: torch.nn.Module,
                device: torch.device):
    model.train()

    loss_value = 0
    y_true_arr, y_pred_arr = None, None

    for batch_items in loader:
        if learning_method == 'supervised':
            x, y_true = batch_items
            assert args.in_channels in [1, 3]
            if args.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x = x.repeat(1, 3, 1, 1)
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)

            loss = loss_fn(y_pred, y_true)
            loss_value += loss.item()

            y_true_np = y_true.detach().cpu().numpy()                        # shape: (batch size, 1)
            y_pred_np = torch.softmax(y_pred, dim=1).detach().cpu().numpy()  # shape: (batch size, num classes)
            if y_true_arr is None:
                y_true_arr = y_true_np
                y_pred_arr = y_pred_np
            else:
                y_true_arr = np.hstack((y_true_arr, y_true_np))
                y_pred_arr = np.vstack((y_pred_arr, y_pred_np))

        else:
            (x_aug1, x_aug2), _ = batch_items
            assert args.in_channels in [1, 3]
            if args.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x_aug1 = x_aug1.repeat(1, 3, 1, 1)
                x_aug2 = x_aug2.repeat(1, 3, 1, 1)
            x_aug1, x_aug2 = x_aug1.to(device), x_aug2.to(device)

            if args.full_grad:
                x_aug1.requires_grad_()
                x_aug2.requires_grad_()

            # Train encoder.
            z1 = model.encode(x_aug1)
            z2 = model.encode(x_aug2)
            p1 = model.project(z1)
            p2 = model.project(z2)

            if args.full_grad:
                grad_z1 = torch.autograd.grad(outputs=z1, inputs=x_aug1, grad_outputs=torch.ones_like(z1), create_graph=True)
                grad_z2 = torch.autograd.grad(outputs=z2, inputs=x_aug2, grad_outputs=torch.ones_like(z2), create_graph=True)
                grad_p1 = torch.autograd.grad(outputs=p1, inputs=x_aug1, grad_outputs=torch.ones_like(p1), create_graph=True)
                grad_p2 = torch.autograd.grad(outputs=p2, inputs=x_aug2, grad_outputs=torch.ones_like(p2), create_graph=True)
                assert len(grad_z1) == len(grad_z2) == len(grad_p1) == len(grad_p2) == 1
                batch_size = x_aug1.shape[0]
                z1 = grad_z1[0].view(batch_size, -1)
                z2 = grad_z2[0].view(batch_size, -1)
                p1 = grad_p1[0].view(batch_size, -1)
                p2 = grad_p2[0].view(batch_size, -1)

            loss = loss_fn(p1_batch=p1, p2_batch=p2, z1_batch=z1, z2_batch=z2)
            loss_value += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr_scheduler.step()

    loss_value /= len(loader)
    if y_true_arr is not None:
        acc = accuracy_score(y_true_arr, np.argmax(y_pred_arr, axis=1))
        auroc = roc_auc_score(y_true_arr, y_pred_arr, average='macro', multi_class='ovo')
        if acc.dtype in [np.float32, np.float64]:
            acc = acc.item()
        if auroc.dtype in [np.float32, np.float64]:
            auroc = auroc.item()
    else:
        acc, auroc = -1, -1  # Placeholder for self-supervised learning.
    return model, optimizer, lr_scheduler, loss_value, acc, auroc


@torch.no_grad()
def infer(model: torch.nn.Module,
          loader: torch.utils.data.DataLoader,
          loss_fn_pred: torch.nn.Module,
          device: torch.device):
    model.eval()

    loss_value = 0
    y_true_arr, y_pred_arr = None, None

    for x, y_true in loader:
        assert args.in_channels in [1, 3]
        if args.in_channels == 1:
            # Repeat the channel dimension: 1 channel -> 3 channels.
            x = x.repeat(1, 3, 1, 1)
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = loss_fn_pred(y_pred, y_true)
        loss_value += loss.item()

        y_true_np = y_true.detach().cpu().numpy()                        # shape: (batch size, 1)
        y_pred_np = torch.softmax(y_pred, dim=1).detach().cpu().numpy()  # shape: (batch size, num classes)
        if y_true_arr is None:
            y_true_arr = y_true_np
            y_pred_arr = y_pred_np
        else:
            y_true_arr = np.hstack((y_true_arr, y_true_np))
            y_pred_arr = np.vstack((y_pred_arr, y_pred_np))

    loss_value /= len(loader)
    acc = accuracy_score(y_true_arr, np.argmax(y_pred_arr, axis=1))
    auroc = roc_auc_score(y_true_arr, y_pred_arr, average='macro', multi_class='ovo')
    if acc.dtype in [np.float32, np.float64]:
        acc = acc.item()
    if auroc.dtype in [np.float32, np.float64]:
        auroc = auroc.item()
    return loss_value, acc, auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet')           # ['resnet', 'resnext', 'convnext', 'vit', 'swin', 'xcit']
    parser.add_argument('--learning-method', type=str, default='simclr') # ['simclr', 'simsiam', 'barlow_twins', 'supervised']
    # parser.add_argument('--mrl', action='store_true')                    # Whether to use Mahalanobis representation learning.
    parser.add_argument('--full-grad', action='store_true')              # Whether to use full gradient for similarity computation.
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs-pretrain', type=int, default=50)
    parser.add_argument('--epochs-tuning', type=int, default=50)
    parser.add_argument('--lr-pretrain', type=float, default=1e-2)
    parser.add_argument('--lr-tuning', type=float, default=1e-4)         # Only relevant to self-supervised learning.
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--dataset-dir', type=str, default='$ROOT_DIR/data/')
    parser.add_argument('--results-dir', type=str, default='$ROOT_DIR/results/')
    args = SimpleNamespace(**vars(parser.parse_args()))

    # Update paths.
    ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
    args.dataset_dir = args.dataset_dir.replace('$ROOT_DIR', ROOT_DIR)
    args.results_dir = args.results_dir.replace('$ROOT_DIR', ROOT_DIR)

    curr_run_identifier = f'dataset-{args.dataset}_model-{args.model}_learning-method-{args.learning_method}_full-grad-{args.full_grad}_PT-epoch-{args.epochs_pretrain}_seed-{args.random_seed}'
    args.log_path = os.path.join(args.results_dir, curr_run_identifier, 'log.txt')
    args.model_pretrain_save_path = os.path.join(args.results_dir, curr_run_identifier, 'model_pretrain.pty')
    args.model_linear_probe_save_path = os.path.join(args.results_dir, curr_run_identifier, 'model_linear_probe.pty')
    args.model_finetune_save_path = os.path.join(args.results_dir, curr_run_identifier, 'model_finetune.pty')
    args.npz_save_path = os.path.join(args.results_dir, curr_run_identifier, 'results.npz')

    main(args)
