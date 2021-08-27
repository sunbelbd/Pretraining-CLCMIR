from copy import deepcopy
from paddle.io import DataLoader, DistributedBatchSampler, RandomSampler, SequenceSampler

from .collate_batch import BatchCollator
from .conceptual_captions_json import ConceptualCaptionsDataset
from .general_corpus import GeneralCorpus
from .parallel_corpus import ParallelCorpus

DATASET_CATALOGS = {'conceptual_captions': ConceptualCaptionsDataset,
                    'general_corpus': GeneralCorpus,
                    'parallel_corpus': ParallelCorpus}


def build_dataset(dataset_name, *args, **kwargs):
    print("Using %s now " % dataset_name)
    assert dataset_name in DATASET_CATALOGS, "dataset not in catalogs"
    return DATASET_CATALOGS[dataset_name](*args, **kwargs)


# def worker_init_fn():
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset
#     worker_id = worker_info.id
#     split_size = len(dataset.data) // worker_info.num_workers
#     dataset.data = dataset.data[worker_id * split_size:(worker_id + 1) * split_size]


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
                                seq_len=cfg.DATASET.SEQ_LEN, min_seq_len=cfg.DATASET.MIN_SEQ_LEN,
                                with_precomputed_visual_feat=cfg.NETWORK.IMAGE_FEAT_PRECOMPUTED,
                                mask_raw_pixels=cfg.NETWORK.MASK_RAW_PIXELS,
                                with_rel_task=cfg.NETWORK.WITH_REL_LOSS,
                                with_mlm_task=cfg.NETWORK.WITH_MLM_LOSS,
                                with_mvrc_task=cfg.NETWORK.WITH_MVRC_LOSS,
                                answer_vocab_file=cfg.DATASET.ANSWER_VOCAB_FILE,
                                root_path=cfg.DATASET.ROOT_PATH, data_path=cfg.DATASET.DATASET_PATH,
                                test_mode=(mode == 'test'),
                                transform=None,
                                zip_mode=cfg.DATASET.ZIP_MODE, cache_mode=cfg.DATASET.CACHE_MODE,
                                cache_db=True if (rank is None or rank == 0) else False,
                                ignore_db_cache=cfg.DATASET.IGNORE_DB_CACHE,
                                add_image_as_a_box=cfg.DATASET.ADD_IMAGE_AS_A_BOX,
                                aspect_grouping=aspect_grouping,
                                mask_size=(cfg.DATASET.MASK_SIZE, cfg.DATASET.MASK_SIZE),
                                pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME)

    # Neither sampler or batch_sampler is compatible with iterable dataset.
    # Check dataset type and decide data loader arguments
    sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank, batch_size)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank)
    collator = BatchCollator(dataset=dataset, append_ind=cfg.DATASET.APPEND_INDEX)
    # if isinstance(dataset, Dataset):
    # dataloader = torch.utils.data.DataLoader(dataset=dataset,
    #                                          batch_size=batch_size,
    #                                          sampler=sampler,
    #                                          num_workers=num_workers,
    #                                          pin_memory=False,
    #                                          collate_fn=collator,
    #                                          drop_last=True)
    dataloader = DataLoader(
        dataset,
        shuffle=False if distributed else shuffle,
        collate_fn=collator,
        batch_sampler=sampler if distributed else None,
        batch_size=1 if distributed else batch_size,
        drop_last=False if distributed else True
    )
    # else:
    #     # save for stream dataset, though not implemented
    #     dataloader = torch.utils.data.DataLoader(dataset=dataset,
    #                                              batch_size=batch_size,
    #                                              num_workers=num_workers,
    #                                              collate_fn=collator,
    #                                              drop_last=True,
    #                                              worker_init_fn=worker_init_fn)
    #     sampler = None

    return dataloader


def make_dataloaders(cfg, mode='train', distributed=False, num_replicas=None, rank=None):
    outputs = []

    for i, dataset_cfg in enumerate(cfg.DATASET):
        cfg_ = deepcopy(cfg)
        cfg_.DATASET = dataset_cfg
        cfg_.TRAIN.BATCH_IMAGES = cfg.TRAIN.BATCH_IMAGES[i]
        cfg_.VAL.BATCH_IMAGES = cfg.VAL.BATCH_IMAGES[i]
        cfg_.TEST.BATCH_IMAGES = cfg.TEST.BATCH_IMAGES[i]
        outputs.append(
            make_dataloader(cfg_,
                            mode=mode,
                            distributed=distributed,
                            num_replicas=num_replicas,
                            rank=rank)
        )

    return outputs
