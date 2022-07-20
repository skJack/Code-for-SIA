from .ffpp import FaceForensics
import torch.utils.data as data
import pdb


def create_dataset(args, split):

    transform = create_data_transforms(args.transform, split)
    base_transform = create_data_transforms_alb(args.transform, split)

    kwargs = getattr(args.dataset, args.dataset.name)
    if args.dataset.name == 'ffpp':
        dataset = FaceForensics(
            split=split,
            base_transform=base_transform,
            transform=transform,
            image_size=args.transform.image_size,
            **kwargs
        )

    else:
        raise Exception('Invalid dataset!')

    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    shuffle = True if sampler is None and split == 'train' else False
    batch_size = getattr(args, split).batch_size
    if args.debug:
        batch_size = 4
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader
