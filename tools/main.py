import argparse
import collections
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
import math



import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from openunreid.apis import BaseRunner, batch_processor, test_reid, set_random_seed
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import (
    build_test_dataloader,
    build_train_dataloader,
    build_val_dataloader,
)
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.models.utils.extract import extract_features, extract_probabilities
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger


from label_stablization import compute_class_stablization_by_entropy, compute_sample_weights, compute_sample_softlabels, compute_label_iou_matrix

class SSBRunner(BaseRunner):
    def update_labels(self):
        sep = "*************************"
        print(f"\n{sep} Start updating pseudo labels on epoch {self._epoch} {sep}\n")
        # generate pseudo labels
        pseudo_labels, label_centers = self.label_generator(
            self._epoch, print_freq=self.print_freq
        )

        if self._epoch == 0:
            self.pseudo_labels_pervious = torch.tensor(pseudo_labels)
            self.pseudo_centers_pervious = label_centers[0].clone().detach().requires_grad_(False)
            self.pervious_label_num = self.pseudo_labels_pervious.unique().size(0)
            self.sample_soft_labels = 0
            self.batch_indexs_labels = 0
            
        else:
            # generate all features
            loaders, datasets = build_val_dataloader(
                self.cfg, for_clustering=True, all_datasets=True
            )
            all_features = []
            for idx, (loader, dataset) in enumerate(zip(loaders, datasets)):
                features = extract_features(
                    self.model, loader, dataset, with_path=False, prefix="Extract: ",
                )
                all_features.append(features)
            del loaders, datasets
            all_features = torch.cat(all_features)
            all_centers = self.pseudo_centers_pervious

            print("ExTrAcTiNg Lables.......................")
            # compute prob labels:
            probs_perv = extract_probabilities(all_features, all_centers, self.cfg.SOFTEN.pred_temp)
            if self._epoch != 1:
                N = probs_perv.size(0) 
                C = probs_perv.size(1)
                onehot_labels = torch.full(size=(N, C), fill_value=0)
                onehot_labels.scatter_(dim=1, index=torch.unsqueeze(self.pseudo_labels_pervious[0], dim=1), value=1)
                probs_perv = self.cfg.SOFTEN.alpha * onehot_labels + (1.0 - self.cfg.SOFTEN.alpha) * probs_perv
 
            self.pseudo_labels_current = torch.tensor(pseudo_labels)
            self.pseudo_centers_current = label_centers[0].clone().detach().requires_grad_(False)
            # compute IOU Mat:
            iou_mat = compute_label_iou_matrix(self.pseudo_labels_pervious, self.pseudo_labels_current)
            norm_iou_mat = (iou_mat.t() / iou_mat.t().sum(0)).t()
            print("COMPUTING SOFTLABELS...")
            all_softlabels = probs_perv.mm(norm_iou_mat)
            print("COMPUTING FINISHED.")


            # Compute Soft Labels:
            beta = self.cfg.SOFTEN.beta
            prob_soft_labels = all_softlabels.cuda()
            hard_iou_labels = compute_sample_softlabels(
                self.pseudo_labels_pervious, self.pseudo_labels_current, "iou", "original"
                ).cuda()
            self.sample_soft_labels = beta * hard_iou_labels + (1.0 - beta) * prob_soft_labels
            self.pseudo_labels_pervious = self.pseudo_labels_current
            self.pseudo_centers_pervious = self.pseudo_centers_current
        synchronize()

        # update train loader
        self.train_loader, self.train_sets = build_train_dataloader(
            self.cfg, pseudo_labels, self.train_sets, self._epoch,
        )

        # re-construct memory
        num_memory = 0
        for idx, set in enumerate(self.train_sets):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                # cluster-level memory for unlabeled data
                num_memory += self.cfg.TRAIN.PSEUDO_LABELS.cluster_num[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
            else:
                # class-level memory for labeled data
                num_memory += set.num_pids

        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            num_features = self.model.module.num_features
        else:
            num_features = self.model.num_features

        self.criterions = build_loss(
            self.cfg.TRAIN.LOSS,
            num_features=num_features,
            num_memory=num_memory,
            cuda=True,
        )

        # initialize memory
        loaders, datasets = build_val_dataloader(
            self.cfg, for_clustering=True, all_datasets=True
        )
        memory_features = []
        for idx, (loader, dataset) in enumerate(zip(loaders, datasets)):
            if idx in cfg.TRAIN.unsup_dataset_indexes:
                memory_features.append(label_centers[cfg.TRAIN.unsup_dataset_indexes.index(idx)])
            else:
                features = extract_features(
                    self.model, loader, dataset, with_path=False, prefix="Extract: ",
                )
                assert features.size(0) == len(dataset)
                centers_dict = collections.defaultdict(list)
                for i, (_, pid, _) in enumerate(dataset):
                    centers_dict[pid].append(features[i].unsqueeze(0))
                centers = [
                    torch.cat(centers_dict[pid], 0).mean(0)
                    for pid in sorted(centers_dict.keys())
                ]
                memory_features.append(torch.stack(centers, 0))
        del loaders, datasets

        memory_features = torch.cat(memory_features)
        self.criterions["hybrid_memory"]._update_feature(memory_features)

        memory_labels = []
        start_pid = 0
        for idx, dataset in enumerate(self.train_sets):
            num_pids = dataset.num_pids
            memory_labels.append(torch.arange(start_pid, start_pid + num_pids))
            start_pid += num_pids
        memory_labels = torch.cat(memory_labels).view(-1)
        self.criterions["hybrid_memory"]._update_label(memory_labels)

        print(f"\n{sep} Finished updating pseudo label {sep}\n")


    def train_step(self, iter, batch):
        # need to be re-written case by case
        assert not isinstance(
            self.model, list
        ), "please re-write 'train_step()' to support list of models"

        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        if len(data["img"]) > 1:
            warnings.warn(
                "please re-write the 'runner.train_step()' function to make use of "
                "mutual transformer."
            )

        inputs = data["img"][0].cuda()
        targets = data["id"].cuda()

        self.batch_indexs_labels = data["ind"].cuda()

        results = self.model(inputs)
        if "prob" in results.keys():
            results["prob"] = results["prob"][
                :, : self.train_loader.loader.dataset.num_pids
            ]

        total_loss = 0
        meters = {}
        
        for key in self.criterions.keys():
            loss = self.criterions[key](results, targets, self._epoch, self.sample_soft_labels, self.batch_indexs_labels, self.cfg.SOFTEN.alpha)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            meters[key] = loss.item()

        if "prob" in results.keys():
            acc = accuracy(results["prob"].data, targets.data)
            meters["Acc@1"] = acc[0]

        self.train_progress.update(meters)

        return total_loss

def parge_config():
    parser = argparse.ArgumentParser(description="Super strong baseline training")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work-dir", help="the dir to save logs and models", default=""
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    if not args.work_dir:
        args.work_dir = Path(args.config).stem
    cfg.work_dir = cfg.LOGS_ROOT / args.work_dir
    mkdir_if_missing(cfg.work_dir)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    shutil.copy(args.config, cfg.work_dir / "config.yaml")

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    synchronize()

    # init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build train loader
    train_loader, train_sets = build_train_dataloader(cfg, joint=False)

    # build model
    model = build_model(cfg, 0, init=cfg.MODEL.source_pretrained)
    model.cuda()
    if dist:
        ddp_cfg = {
            "device_ids": [cfg.gpu],
            "output_device": cfg.gpu,
            "find_unused_parameters": True,
        }
        model = DistributedDataParallel(model, **ddp_cfg)
    elif cfg.total_gpus > 1:
        model = DataParallel(model)

    # build optimizer
    optimizer = build_optimizer([model], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler = None

    # build loss functions
    num_memory = 0
    for idx, set in enumerate(train_sets):
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # instance-level memory for unlabeled data
            num_memory += len(set)
        else:
            # class-level memory for labeled data
            num_memory += set.num_pids

    if isinstance(model, (DataParallel, DistributedDataParallel)):
        num_features = model.module.num_features
    else:
        num_features = model.num_features

    criterions = build_loss(
        cfg.TRAIN.LOSS,
        num_features=num_features,
        num_memory=num_memory,
        cuda=True,
    )

    # build runner
    runner = SSBRunner(
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        train_sets=train_sets,
        lr_scheduler=lr_scheduler,
        meter_formats={"Time": ":.3f",},
        reset_optim=False,
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    # start training
    runner.run()

    # load the best model
    runner.resume(cfg.work_dir / "model_best.pth")

    # final testing
    test_loaders, queries, galleries = build_test_dataloader(cfg)
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
        cmc, mAP = test_reid(
            cfg, model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i]
        )

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()
