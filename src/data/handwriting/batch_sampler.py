from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import Sampler


class BatchSampler(Sampler):

    def __init__(self, dataset, txt_batch_size, img_batch_size, img_txt_batch_size, iter_dataset_index):

        self.dataset = dataset

        self.txt_batch_size = txt_batch_size
        self.img_batch_size = img_batch_size
        self.img_txt_batch_size = img_txt_batch_size
        self.batch_size = self.txt_batch_size + self.img_batch_size + self.img_txt_batch_size

        self.number_of_datasets = len(dataset.datasets)
        self.dataset_size = [len(cur_dataset) for cur_dataset in dataset.datasets]
        self.iter_dataset_index = iter_dataset_index

    def __len__(self):
        return self.dataset_size[self.iter_dataset_index] * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []

        for idx in range(self.number_of_datasets):
            current_dataset = self.dataset.datasets[idx]
            sampler = RandomSampler(current_dataset)
            samplers_list.append(sampler)
            current_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(current_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]

        samples_to_grab = [self.txt_batch_size, self.img_batch_size, self.img_txt_batch_size]

        epoch_samples = self.dataset_size[self.iter_dataset_index]
        step = samples_to_grab[self.iter_dataset_index]

        result = []

        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):

                current_batch_sampler = sampler_iterators[i]
                current_samples = []

                for _ in range(samples_to_grab[i]):

                    try:
                        current_sample_org = current_batch_sampler.__next__()

                    except StopIteration:
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        current_batch_sampler = sampler_iterators[i]
                        current_sample_org = current_batch_sampler.__next__()

                    current_sample = current_sample_org + push_index_val[i]
                    current_samples.append(current_sample)

                result.extend(current_samples)

        result = iter(result)

        del samplers_list
        del sampler_iterators

        return result
