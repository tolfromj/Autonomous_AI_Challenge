from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import AutoImageProcessor   ################## 나중에 수정하기 #######

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    https://github.com/victoresque/pytorch-template/blob/master/base/base_data_loader.py#L7
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split=0.0, num_workers=1, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        
        if collate_fn == 'detr':
            self.collate_fn = self._collate_fn
        else: self.collate_fn = collate_fn
            
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def _collate_fn(self, batch):
        CHECKPOINT = self.get_model_checkpoint()
        image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT) ################## 나중에 수정하기 #######
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        # batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch

    def get_model_checkpoint(self):  
        return "facebook/detr-resnet-50"  ################## 나중에 수정하기 #######