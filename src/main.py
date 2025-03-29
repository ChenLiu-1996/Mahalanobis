from types import SimpleNamespace
import argparse
import os
import sys
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

import numpy as np
import torch
import torchvision
from tinyimagenet import TinyImageNet
from tqdm import tqdm


import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/model/')
from timm_models import build_timm_model
sys.path.insert(0, import_dir + '/src/nn_utils/')
from log import log
from seed import seed_everything
from simclr import SingleInstanceTwoView
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

    if args.loss_fn == 'supervised':
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

    elif args.loss_fn in ['simclr', 'simsiam', 'barlow_twins']:
        transform_train = SingleInstanceTwoView(imsize=imsize,
                                                mean=dataset_mean,
                                                std=dataset_std)

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            imsize,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(imsize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    if args.dataset in ['mnist', 'cifar10', 'cifar100']:
        train_dataset = torchvision_dataset(args.dataset_dir,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
        val_dataset = torchvision_dataset(args.dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_val)

    elif args.dataset in ['stanfordcars', 'stl10', 'food101', 'flowers102']:
        train_dataset = torchvision_dataset(args.dataset_dir,
                                            split='train',
                                            download=True,
                                            transform=transform_train)
        val_dataset = torchvision_dataset(args.dataset_dir,
                                          split='test',
                                          download=True,
                                          transform=transform_val)

        if args.dataset == 'stl10':
            # Training set has too few images (5000 images in total).
            # Let's augment it into a bigger dataset.
            train_dataset = ExtendedDataset(train_dataset,
                                            desired_len=10 *
                                            len(train_dataset))

    elif args.dataset in ['tinyimagenet', 'imagenet']:
        train_dataset = torchvision_dataset(args.dataset_dir,
                                            split='train',
                                            transform=transform_train)
        val_dataset = torchvision_dataset(args.dataset_dir,
                                          split='val',
                                          transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False,
                                             pin_memory=True)

    return (train_loader, val_loader), args


def simsiam(z1_batch: torch.Tensor,
            z2_batch: torch.Tensor,
            p1_batch: torch.Tensor,
            p2_batch: torch.Tensor) -> torch.Tensor:
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
           temperature: float = 0.5) -> torch.Tensor:
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
                 lambda_off_diag: float = 1e-2) -> torch.Tensor:
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


def train(args: SimpleNamespace) -> None:
    '''
    The main function of training and evaluation.
    '''
    # Log the config.

    config_str = 'Config: \n'
    for key in args.keys():
        config_str += '%s: %s\n' % (key, args[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=args.log_path, to_console=False)

    seed_everything(args.random_seed)

    model = build_timm_model(model_name=args.model,
                             num_classes=args.num_classes).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloaders, args = get_dataloaders(args=args)
    train_loader, val_loader = dataloaders

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    # Loss function.
    loss_fn_pred = torch.nn.CrossEntropyLoss()
    if args.loss_fn == 'supervised':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif args.loss_fn == 'simclr':
        loss_fn = simclr
    elif args.loss_fn == 'simsiam':
        loss_fn = simsiam
    elif args.loss_fn == 'barlow_twins':
        loss_fn = barlow_twins
    else:
        raise ValueError(f'loss function `{args.loss_fn}` not supported.')

    val_metric = 'val_auroc'

    # Compute the results before training.
    val_loss, val_acc, val_auroc = infer(loader=val_loader, model=model, loss_fn_pred=loss_fn_pred, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))

    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                                 warmup_epochs=min(20, args.epochs_pretrain),
                                                 warmup_start_lr=args.learning_rate_pretrain * 1e-2,
                                                 max_epochs=args.epochs_pretrain)


    best_val_metric = 0
    best_model = None

    # val_metric_pct_list = [20, 30, 40, 50, 60, 70, 80, 90]
    # is_model_saved = {}
    # for val_metric_pct in val_metric_pct_list:
    #     is_model_saved[str(val_metric_pct)] = False

    for epoch_idx in tqdm(range(1, args.max_epoch)):
        # For SimCLR, only perform validation / linear probing every 5 epochs.
        skip_epoch_simlr = epoch_idx % 5 != 0

        state_dict = {
            'train_loss': 0,
            'train_acc': 0,
            'val_loss': 0,
            'val_acc': 0,
            'acc_diverg': 0,
        }

        if args.loss_fn == 'simclr':
            state_dict['train_simclr_pseudoAcc'] = 0

        #
        '''
        Training
        '''
        model.train()
        # Because of linear warmup, first step has zero LR. Hence step once before training.
        lr_scheduler.step()
        correct, total_count_loss, total_count_acc = 0, 0, 0
        for _, (x, y_true) in enumerate(tqdm(train_loader)):
            if args.loss_fn in ['supervised', 'wronglabel']:
                # Not using contrastive learning.

                B = x.shape[0]
                assert args.in_channels in [1, 3]
                if args.in_channels == 1:
                    # Repeat the channel dimension: 1 channel -> 3 channels.
                    x = x.repeat(1, 3, 1, 1)
                x, y_true = x.to(device), y_true.to(device)

                y_pred = model(x)
                loss = loss_fn(y_pred, y_true)
                state_dict['train_loss'] += loss.item() * B
                correct += torch.sum(
                    torch.argmax(y_pred, dim=-1) == y_true).item()
                total_count_loss += B
                total_count_acc += B

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif args.loss_fn == 'simclr':
                # Using SimCLR.

                x_aug1, x_aug2 = x
                B = x_aug1.shape[0]
                assert args.in_channels in [1, 3]
                if args.in_channels == 1:
                    # Repeat the channel dimension: 1 channel -> 3 channels.
                    x_aug1 = x_aug1.repeat(1, 3, 1, 1)
                    x_aug2 = x_aug2.repeat(1, 3, 1, 1)
                x_aug1, x_aug2, y_true = x_aug1.to(device), x_aug2.to(
                    device), y_true.to(device)

                # Train encoder.
                z1 = model.project(x_aug1)
                z2 = model.project(x_aug2)

                loss, pseudo_acc = loss_fn_simclr(z1, z2)
                state_dict['train_loss'] += loss.item() * B
                state_dict['train_simclr_pseudoAcc'] += pseudo_acc * B
                total_count_loss += B

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if args.loss_fn == 'simclr':
            state_dict['train_simclr_pseudoAcc'] /= total_count_loss
        else:
            state_dict['train_acc'] = correct / total_count_acc * 100
        state_dict['train_loss'] /= total_count_loss

        #
        '''
        Validation (or Linear Probing + Validation)
        '''
        if args.loss_fn == 'simclr':
            if not skip_epoch_simlr:
                # This function call includes validation.
                probing_acc, val_acc_final, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, \
                    dsmi_blockZ_Xs, dsmi_blockZ_Ys, _ = linear_probing(
                    config=args,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    device=device,
                    loss_fn_classification=loss_fn,
                    precomputed_clusters_X=precomputed_clusters_X)
                state_dict['train_acc'] = probing_acc
                state_dict['val_loss'] = np.nan
                state_dict['val_acc'] = val_acc_final
            else:
                state_dict['train_acc'] = 'Val skipped for efficiency'
                state_dict['val_loss'] = 'Val skipped for efficiency'
                state_dict['val_acc'] = 'Val skipped for efficiency'
        else:
            val_loss, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, \
                dsmi_blockZ_Xs, dsmi_blockZ_Ys, _ = validate_epoch(
                args=args,
                val_loader=val_loader,
                model=model,
                device=device,
                loss_fn=loss_fn,
                precomputed_clusters_X=precomputed_clusters_X)
            state_dict['val_loss'] = val_loss
            state_dict['val_acc'] = val_acc

        if not (args.loss_fn == 'simclr' and skip_epoch_simlr):
            state_dict['acc_diverg'] = \
                state_dict['train_acc'] - state_dict['val_acc']
        else:
            state_dict['acc_diverg'] = 'Val skipped for efficiency'

        log('Epoch: %d. %s' % (epoch_idx, print_state_dict(state_dict)),
            filepath=log_path,
            to_console=False)

        if not (args.loss_fn == 'simclr' and skip_epoch_simlr):
            results_dict['epoch'].append(epoch_idx)
            results_dict['dse_Z'].append(dse_Z)
            results_dict['cse_Z'].append(cse_Z)
            results_dict['dsmi_Z_X'].append(dsmi_Z_X)
            results_dict['csmi_Z_X'].append(csmi_Z_X)
            results_dict['dsmi_Z_Y'].append(dsmi_Z_Y)
            results_dict['csmi_Z_Y'].append(csmi_Z_Y)
            results_dict['val_acc'].append(state_dict['val_acc'])
            results_dict['dsmi_blockZ_Xs'].append(np.array(dsmi_blockZ_Xs))
            results_dict['dsmi_blockZ_Ys'].append(np.array(dsmi_blockZ_Ys))

        # Save best model
        if not (args.loss_fn == 'simclr' and skip_epoch_simlr):
            if state_dict[val_metric] > best_val_metric:
                best_val_metric = state_dict[val_metric]
                best_model = model.state_dict()
                model_save_path = '%s/%s-%s-%s-ConvInitStd-%s-seed%s-%s' % (
                    args.checkpoint_dir, args.dataset, args.loss_fn,
                    args.model, args.conv_init_std, args.random_seed,
                    '%s_best.pth' % val_metric)
                torch.save(best_model, model_save_path)
                log('Best model (so far) successfully saved.',
                    filepath=log_path,
                    to_console=False)

            model_save_path = '%s/%s-%s-%s-ConvInitStd-%s-seed%s-epoch-%s.pth' % (
                args.checkpoint_dir, args.dataset, args.loss_fn,
                args.model, args.conv_init_std, args.random_seed,
                epoch_idx)
            torch.save(model.state_dict(), model_save_path)

        if epoch_idx > 30:
            break

    # Save the results after training.
    save_path_numpy = '%s/%s-%s-%s-ConvInitStd-%s-seed%s/%s' % (
        args.output_save_path, args.dataset, args.loss_fn, args.model,
        args.conv_init_std, args.random_seed, 'results.npz')
    os.makedirs(os.path.dirname(save_path_numpy), exist_ok=True)

    with open(save_path_numpy, 'wb+') as f:
        np.savez(
            f,
            epoch=np.array(results_dict['epoch']),
            val_acc=np.array(results_dict['val_acc']),
            dse_Z=np.array(results_dict['dse_Z']),
            cse_Z=np.array(results_dict['cse_Z']),
            dsmi_Z_X=np.array(results_dict['dsmi_Z_X']),
            csmi_Z_X=np.array(results_dict['csmi_Z_X']),
            dsmi_Z_Y=np.array(results_dict['dsmi_Z_Y']),
            csmi_Z_Y=np.array(results_dict['csmi_Z_Y']),
        )

    # Save block by block DSMI results
    save_path_numpy = '%s/%s-%s-%s-ConvInitStd-%s-seed%s/%s' % (
        args.output_save_path, args.dataset, args.loss_fn, args.model,
        args.conv_init_std, args.random_seed, 'block-results.npz')
    os.makedirs(os.path.dirname(save_path_numpy), exist_ok=True)

    with open(save_path_numpy, 'wb+') as f:
        np.savez(
            f,
            epoch=np.array(results_dict['epoch']),
            dsmi_blockZ_Xs=results_dict['dsmi_blockZ_Xs'],
            dsmi_blockZ_Ys=results_dict['dsmi_blockZ_Ys'],
        )

    return


@torch.no_grad()
def infer(loader, model, loss_fn_pred, device):
    avg_loss = 0
    y_true_arr, y_pred_arr = None, None

    for x, y_true in loader:
        x = x.to(device)
        y_pred = model(x)
        loss = loss_fn_pred(y_pred, y_true.to(device))
        avg_loss += loss.item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    avg_loss /= len(loader)
    acc = np.mean(accuracy_score(y_true_arr, y_pred_arr))
    auroc = np.mean(roc_auc_score(y_true_arr, y_pred_arr))
    return avg_loss, acc, auroc


def linear_probing(config: SimpleNamespace,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module, device: torch.device,
                   loss_fn_classification: torch.nn.Module,
                   precomputed_clusters_X: np.array):

    # Separately train linear classifier.
    model.init_linear()
    # Note: Need to create another optimizer because the model will keep updating
    # even after freezing with `requires_grad = False` when `opt` has `momentum`.
    opt_probing = torch.optim.AdamW(list(model.linear.parameters()),
                                    lr=float(config.learning_rate_probing))

    lr_scheduler_probing = LinearWarmupCosineAnnealingLR(
        optimizer=opt_probing,
        warmup_epochs=min(10, config.probing_epoch // 5),
        max_epochs=config.probing_epoch)

    for _ in tqdm(range(config.probing_epoch)):
        # Because of linear warmup, first step has zero LR. Hence step once before training.
        lr_scheduler_probing.step()
        probing_acc = linear_probing_epoch(
            config=config,
            train_loader=train_loader,
            model=model,
            device=device,
            opt_probing=opt_probing,
            loss_fn_classification=loss_fn_classification)

    _, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, dsmi_blockZ_Xs, dsmi_blockZ_Ys, _ = validate_epoch(
        args=config,
        val_loader=val_loader,
        model=model,
        device=device,
        loss_fn=loss_fn_classification,
        precomputed_clusters_X=precomputed_clusters_X)

    return probing_acc, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, dsmi_blockZ_Xs, dsmi_blockZ_Ys, precomputed_clusters_X


def linear_probing_epoch(config: SimpleNamespace,
                         train_loader: torch.utils.data.DataLoader,
                         model: torch.nn.Module, device: torch.device,
                         opt_probing: torch.optim.Optimizer,
                         loss_fn_classification: torch.nn.Module):
    model.train()
    correct, total_count_acc = 0, 0
    for _, (x, y_true) in enumerate(train_loader):
        x_aug1, x_aug2 = x
        B = x_aug1.shape[0]
        assert config.in_channels in [1, 3]
        if config.in_channels == 1:
            # Repeat the channel dimension: 1 channel -> 3 channels.
            x_aug1 = x_aug1.repeat(1, 3, 1, 1)
            x_aug2 = x_aug2.repeat(1, 3, 1, 1)
        x_aug1, x_aug2, y_true = x_aug1.to(device), x_aug2.to(
            device), y_true.to(device)

        with torch.no_grad():
            h1, h2 = model.encode(x_aug1), model.encode(x_aug2)
        y_pred_aug1, y_pred_aug2 = model.linear(h1), model.linear(h2)
        loss_aug1 = loss_fn_classification(y_pred_aug1, y_true)
        loss_aug2 = loss_fn_classification(y_pred_aug2, y_true)
        loss = (loss_aug1 + loss_aug2) / 2
        correct += torch.sum(
            torch.argmax(y_pred_aug1, dim=-1) == y_true).item()
        correct += torch.sum(
            torch.argmax(y_pred_aug2, dim=-1) == y_true).item()
        total_count_acc += 2 * B

        opt_probing.zero_grad()
        loss.backward()
        opt_probing.step()

    probing_acc = correct / total_count_acc * 100

    return probing_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet')   # ['resnet', 'resnext', 'convnext', 'vit', 'swin', 'xcit']
    parser.add_argument('--loss-fn', type=str, default='simclr') # ['simclr', 'simsiam', 'barlow_twins', 'supervised']
    parser.add_argument('--mrl', action='store_true')            # Whether to use Mahalanobis representation learning.
    parser.add_argument('--full-grad', action='store_true')      # Whether to use full gradient for similarity computation.
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs-pretrain', type=int, default=50)
    parser.add_argument('--epochs-finetune', type=int, default=50)
    parser.add_argument('--lr-pretrain', type=float, default=1e-2)
    parser.add_argument('--lr-finetune', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--dataset-dir', type=str, default='$ROOT_DIR/data/')
    parser.add_argument('--results-dir', type=str, default='$ROOT_DIR/results/')
    args = SimpleNamespace(**vars(parser.parse_args()))

    # Update paths.
    ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
    args.dataset_dir = args.dataset_dir.replace('$ROOT_DIR', ROOT_DIR)
    args.results_dir = args.results_dir.replace('$ROOT_DIR', ROOT_DIR)

    curr_run_identifier = f'dataset-{args.dataset}_model-{args.model}_loss-fn-{args.loss_fn}_mrl-{args.mrl}_full-grad-{args.full_grad}_seed-{args.random_seed}'
    args.log_path = os.path.join(args.results_dir, curr_run_identifier, 'log.txt')
    args.model_save_path = os.path.join(args.results_dir, curr_run_identifier, 'model.pty')

    train(args)
