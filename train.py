import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

from config import get_config
from image_iter import FaceDataset

from util.utils import (
    separate_irse_bn_paras,
    separate_resnet_bn_paras,
    separate_mobilefacenet_bn_paras,
)
from util.utils import (
    get_val_data,
    perform_val,
    get_time,
    buffer_val,
    AverageMeter,
    train_accuracy,
)

import time
from vit_pytorch import ViT_face
from vit_pytorch import ViTs_face
from vit_pytorch import NAT  # Imported the NAT

# from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer


# ======= Added epoch_change boolean value, such that the checkpoint is saved when a new epoch starts =======#
def need_save(acc, highest_acc, is_epoch_change):
    if is_epoch_change:
        return True
    do_save = False
    save_cnt = 0
    if acc[0] > 0.49:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i] - 0.002:
            save_cnt += 1
    if save_cnt >= len(acc) * 3 / 4 and acc[0] > 0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="for face verification",
    )
    parser.add_argument(
        "-w",
        "--workers_id",
        help="gpu ids or cpu ['0', '1', '2', '3'] (default: )",
        default="cpu",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="training epochs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="batch_size",
        default=256,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--data_mode",
        help="use which database, [casia, vgg, ms1m, retina, ms1mr]",
        default="ms1m",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--net",
        help="which network, ['VIT','VITs','SWT','NAT]",
        default="VITs",
        type=str,
    )
    parser.add_argument(
        "-head",
        "--head",
        help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']",
        default="ArcFace",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--target",
        help="verification targets",
        default="lfw",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="resume model",
        default="",
        type=str,
    )
    parser.add_argument(
        "--outdir",
        help="output dir",
        default="",
        type=str,
    )
    parser.add_argument(
        "--model",
        help="model name for nat",
        default="nat_mini",
        type=str,
    )
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw")',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="weight decay (default: 0.05)",
    )

    # ======= Learning rate schedule parameters =======#
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine")',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10)",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )
    args = parser.parse_args()

    # ======= Hyperparameters & Data Loaders =======#
    cfg = get_config(args)

    SEED = cfg["SEED"]  # Random Seed for Reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg[
        "DATA_ROOT"
    ]  # The parent root where your train/val/test data are stored
    EVAL_PATH = cfg["EVAL_PATH"]
    WORK_PATH = cfg[
        "WORK_PATH"
    ]  # The root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg[
        "BACKBONE_RESUME_ROOT"
    ]  # The root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg["BACKBONE_NAME"]
    HEAD_NAME = cfg[
        "HEAD_NAME"
    ]  # Support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']

    INPUT_SIZE = cfg["INPUT_SIZE"]
    EMBEDDING_SIZE = cfg["EMBEDDING_SIZE"]  # Feature Dimension
    BATCH_SIZE = cfg["BATCH_SIZE"]
    NUM_EPOCH = 125

    DEVICE = cfg["DEVICE"]
    MULTI_GPU = cfg["MULTI_GPU"]  # Flag to use multiple GPUs
    GPU_ID = cfg["GPU_ID"]  # Specify GPU ids
    print("GPU_ID", GPU_ID)
    TARGET = cfg["TARGET"]
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, "config.txt"), "w") as f:
        f.write(str(cfg))
    print("=" * 60)

    writer = SummaryWriter(WORK_PATH)  # Writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True

    with open(os.path.join(DATA_ROOT, "property"), "r") as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(",")]
    assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]

    dataset = FaceDataset(os.path.join(DATA_ROOT, "train.rec"), rand_mirror=True)
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=len(GPU_ID),
        drop_last=True,
    )

    print("Number of Training Classes: {}".format(NUM_CLASS))

    vers = get_val_data(EVAL_PATH, TARGET)
    highest_acc = [0.0 for t in TARGET]

    # embed()

    # ======= Model, Loss & Optimizer =======#
    BACKBONE_DICT = {
        "VIT": ViT_face(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=16,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        ),
        "VITs": ViTs_face(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=16,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        ),
        # Used the NAT model. Used the nat_mini model with the required hyperparameters.
        "NAT": NAT(
            depths=[3, 4, 18, 5],
            num_heads=[2, 4, 8, 16],
            mlp_ratio=3,
            embed_dim=64,
            drop_path_rate=0.2,
            kernel_size=7,
            num_classes=NUM_CLASS,
        ),
    }

    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]

    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    LOSS = nn.CrossEntropyLoss()

    # embed()

    OPTIMIZER = create_optimizer(args, BACKBONE)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)
    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)

    epoch = 0  # Setting Epoch

    # Multi-GPU setting
    if MULTI_GPU:
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)

    # Single-GPU setting
    else:
        BACKBONE = BACKBONE.to(DEVICE)

    INITIAL = -1
    batch = 0  # Batch Index

    # Optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)

        """
        Loaded the checkpoint parameters, model state dictionary, optimizer dictionary, epoch, loss, batch
        """

        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            checkpoint = torch.load(
                BACKBONE_RESUME_ROOT,
                map_location=DEVICE,
            )
            BACKBONE.load_state_dict(checkpoint["model_state_dict"])
            OPTIMIZER.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            INITIAL = checkpoint["epoch"]
            LOSS = checkpoint["loss"]
            BATCH = checkpoint["batch"]
        else:
            print(
                "No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(
                    BACKBONE_RESUME_ROOT
                )
            )
        print("=" * 60)

    # ======= Train, Validation & Save checkpoint =======#
    DISP_FREQ = 10  # Frequency to display training loss & accuracy
    VER_FREQ = 100

    losses = AverageMeter()
    top1 = AverageMeter()

    batches = len(trainloader)

    BACKBONE.train()  # Set to training mode

    # ======= The epoch starts from the checkpoint epoch =======#
    while epoch < NUM_EPOCH:  # Start Training process
        lr_scheduler.step(epoch)

        last_time = time.time()

        for inputs, labels in iter(trainloader):
            if INITIAL == epoch and batch <= BATCH:
                batch += 1
                continue

            # ======= Compute output =======#
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()

            """
            The NAT backbone only requires the input values, thus when the backbone is NAT, pass the inputs, else the labels and the inputs is passed.
            """

            if BACKBONE_NAME == "NAT":
                outputs = BACKBONE(inputs.float())
            else:
                outputs, emb = BACKBONE(inputs.float(), labels)

            # print(outputs.shape)

            loss = LOSS(outputs, labels)

            # print("outputs", outputs, outputs.data)

            # Measure accuracy and record loss =======#
            prec1 = train_accuracy(outputs.data, labels, topk=(1,))

            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))

            # ======= Compute Gradient Descent & do SGD step =======#
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            # ======= Display training loss & acc every DISP_FREQ (buffer for visualization) =======#
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                epoch_loss = losses.avg
                epoch_acc = top1.avg

                writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)

                batch_time = time.time() - last_time
                last_time = time.time()

                print(
                    "Epoch {} Batch {}\t"
                    "Speed: {speed:.2f} samples/s\t"
                    "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch + 1,
                        batch + 1,
                        speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                        loss=losses,
                        top1=top1,
                    )
                )

                # print("=" * 60)
                losses = AverageMeter()
                top1 = AverageMeter()

            # ======= Added another condition that when epoch changes i.e batch % number_of_batches is 0 =======#
            if (
                ((batch + 1) % VER_FREQ == 0) or batch % batches == 0
            ) and batch != 0:  # Perform Validation & Save checkpoints (Buffer for Visualization)
                for params in OPTIMIZER.param_groups:
                    lr = params["lr"]
                    break
                print("Learning rate %f" % lr)
                print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
                acc = []
                for ver in vers:
                    name, data_set, issame = ver
                    accuracy, std, xnorm, best_threshold, roc_curve = perform_val(
                        MULTI_GPU,
                        DEVICE,
                        EMBEDDING_SIZE,
                        BATCH_SIZE,
                        BACKBONE,
                        data_set,
                        issame,
                    )
                    buffer_val(
                        writer,
                        name,
                        accuracy,
                        std,
                        xnorm,
                        best_threshold,
                        roc_curve,
                        batch + 1,
                    )
                    print("[%s][%d]XNorm: %1.5f" % (name, batch + 1, xnorm))
                    print(
                        "[%s][%d]Accuracy-Flip: %1.5f+-%1.5f"
                        % (name, batch + 1, accuracy, std)
                    )
                    print(
                        "[%s][%d]Best-Threshold: %1.5f"
                        % (name, batch + 1, best_threshold)
                    )
                    acc.append(accuracy)

                is_epoch_change = False
                if batch % batches == 0:
                    is_epoch_change = True

                # ======= Save checkpoints per epoch =======#
                if need_save(acc, highest_acc, is_epoch_change):
                    if is_epoch_change:
                        print("Saving on Epoch change...")
                        print(f"After Epoch {epoch}")

                        """ While Saving, saved epoch, optimizer, model, loss, and batch """

                    if MULTI_GPU:
                        torch.save(
                            BACKBONE.module.state_dict(),
                            os.path.join(
                                WORK_PATH,
                                "Backbone_{}_checkpoint.pth".format(
                                    BACKBONE_NAME,
                                ),
                            ),
                        )
                    else:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": BACKBONE.state_dict(),
                                "optimizer_state_dict": OPTIMIZER.state_dict(),
                                "loss": LOSS,
                                "batch": batch,
                            },
                            os.path.join(
                                WORK_PATH,
                                "Backbone_{}_LR_checkpoint.pth".format(BACKBONE_NAME),
                            ),
                        )

                BACKBONE.train()  # Set to Training mode

            batch += 1  # Batch Index

        epoch += 1
