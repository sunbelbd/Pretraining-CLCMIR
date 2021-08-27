from paddle.io import DataLoader, DistributedBatchSampler, RandomSampler, SequenceSampler

from .collate_batch import BatchCollator
from .xlretcoco import XLRETCOCODataset

DATASET_CATALOGS = {'xlretcoco': XLRETCOCODataset}


def build_dataset(dataset_name, *args, **kwargs):
    assert dataset_name in DATASET_CATALOGS, "dataset not in catalogs"
    return DATASET_CATALOGS[dataset_name](*args, **kwargs)


def make_data_sampler(dataset, shuffle, distributed, num_replicas, rank, batch_size):
    if distributed:
        return DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequenceSampler(dataset)
    return sampler


def make_dataloader(cfg, dataset=None, mode='train', distributed=False, num_replicas=None, rank=None):
    assert mode in ['train', 'val', 'test']
    if mode == 'train':
        ann_file = cfg.DATASET.TRAIN_ANNOTATION_FILE
        image_set = cfg.DATASET.TRAIN_IMAGE_SET
        aspect_grouping = cfg.TRAIN.ASPECT_GROUPING
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TRAIN.BATCH_IMAGES * num_gpu
        shuffle = cfg.TRAIN.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
    elif mode == 'val':
        ann_file = cfg.DATASET.VAL_ANNOTATION_FILE
        image_set = cfg.DATASET.VAL_IMAGE_SET
        aspect_grouping = False
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.VAL.BATCH_IMAGES * num_gpu
        shuffle = cfg.VAL.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
    else:
        ann_file = cfg.DATASET.TEST_ANNOTATION_FILE
        image_set = cfg.DATASET.TEST_IMAGE_SET
        aspect_grouping = False
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TEST.BATCH_IMAGES * num_gpu
        shuffle = cfg.TEST.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu

    # transform = build_transforms(cfg, mode)

    if dataset is None:
        dataset = build_dataset(dataset_name=cfg.DATASET.DATASET, ann_file=ann_file, image_set=image_set,
                                seq_len=cfg.DATASET.SEQ_LEN,
                                with_precomputed_visual_feat=cfg.NETWORK.IMAGE_FEAT_PRECOMPUTED,
                                root_path=cfg.DATASET.ROOT_PATH, data_path=cfg.DATASET.DATASET_PATH,
                                test_mode=(mode != 'train'),
                                transform=None,
                                cache_mode=cfg.DATASET.CACHE_MODE,
                                cache_db=True if (rank is None or rank == 0) else False,
                                add_image_as_a_box=cfg.DATASET.ADD_IMAGE_AS_A_BOX,
                                aspect_grouping=aspect_grouping,
                                pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME)

    collator = BatchCollator(dataset=dataset, append_ind=cfg.DATASET.APPEND_INDEX)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas,
    #                                                          rank=rank)
    sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank, batch_size)
    dataloader = DataLoader(
        dataset,
        shuffle=False if distributed else shuffle,
        collate_fn=collator,
        batch_sampler=sampler if distributed else None,
        batch_size=1 if distributed else batch_size,
        drop_last=False if distributed else True
    )

    return dataloader
