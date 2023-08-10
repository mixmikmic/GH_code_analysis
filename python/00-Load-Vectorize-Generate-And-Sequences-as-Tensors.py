import json

from local_settings import settings, datautils

from datautils.vocabulary import Vocabulary

import pandas as pd
import numpy as np

import torch
from torch import FloatTensor
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm, tqdm_notebook

class RawSurnames(object):
    def __init__(self, data_path=settings.SURNAMES_CSV, delimiter=","):
        self.data = pd.read_csv(data_path, delimiter=delimiter)

    def get_data(self, filter_to_nationality=None):
        if filter_to_nationality is not None:
            return self.data[self.data.nationality.isin(filter_to_nationality)]
        return self.data

class VectorizedSurnames(Dataset):
    def __init__(self, x_surnames, y_nationalities):
        self.x_surnames = x_surnames
        self.y_nationalities = y_nationalities

    def __len__(self):
        return len(self.x_surnames)

    def __getitem__(self, index):
        return {'x_surnames': self.x_surnames[index],
                'y_nationalities': self.y_nationalities[index],
                'x_lengths': len(self.x_surnames[index].nonzero()[0])}


class SurnamesVectorizer(object):
    def __init__(self, surname_vocab, nationality_vocab, max_seq_length):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        self.max_seq_length = max_seq_length
        
    def save(self, filename):
        vec_dict = {"surname_vocab": self.surname_vocab.get_serializable_contents(),
                    "nationality_vocab": self.nationality_vocab.get_serializable_contents(),
                    'max_seq_length': self.max_seq_length}

        with open(filename, "w") as fp:
            json.dump(vec_dict, fp)
        
    @classmethod
    def load(cls, filename):
        with open(filename, "r") as fp:
            vec_dict = json.load(fp)

        vec_dict["surname_vocab"] = Vocabulary.deserialize_from_contents(vec_dict["surname_vocab"])
        vec_dict["nationality_vocab"] = Vocabulary.deserialize_from_contents(vec_dict["nationality_vocab"])
        return cls(**vec_dict)

    @classmethod
    def fit(cls, surname_df):
        """
        """
        surname_vocab = Vocabulary(use_unks=False,
                                   use_mask=True,
                                   use_start_end=True,
                                   start_token=settings.START_TOKEN,
                                   end_token=settings.END_TOKEN)

        nationality_vocab = Vocabulary(use_unks=False, use_start_end=False, use_mask=False)

        max_seq_length = 0
        for index, row in surname_df.iterrows():
            surname_vocab.add_many(row.surname)
            nationality_vocab.add(row.nationality)

            if len(row.surname) > max_seq_length:
                max_seq_length = len(row.surname)
        max_seq_length = max_seq_length + 2

        return cls(surname_vocab, nationality_vocab, max_seq_length)

    @classmethod
    def fit_transform(cls, surname_df, split='train'):
        vectorizer = cls.fit(surname_df)
        return vectorizer, vectorizer.transform(surname_df, split)

    def transform(self, surname_df, split='train'):

        df = surname_df[surname_df.split==split].reset_index()
        n_data = len(df)
        
        x_surnames = np.zeros((n_data, self.max_seq_length), dtype=np.int64)
        y_nationalities = np.zeros(n_data, dtype=np.int64)

        for index, row in df.iterrows():
            vectorized_surname = list(self.surname_vocab.map(row.surname, 
                                                             include_start_end=True))
            x_surnames[index, :len(vectorized_surname)] = vectorized_surname
            y_nationalities[index] = self.nationality_vocab[row.nationality]

        return VectorizedSurnames(x_surnames, y_nationalities)

# data generator

def make_generator(vectorized_data, batch_size, num_batches=-1, 
                               num_workers=0, volatile_mode=False, 
                               strict_batching=True):

    loaded_data = DataLoader(vectorized_data, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)

    def inner_func(num_batches=num_batches, 
                   volatile_mode=volatile_mode):

        for batch_index, batch in enumerate(loaded_data):
            out = {}
            current_batch_size = list(batch.values())[0].size(0)
            if current_batch_size < batch_size and strict_batching:
                break
            for key, value in batch.items():
                if not isinstance(value, Variable):
                    value = Variable(value)
                if settings.CUDA:
                    value = value.cuda()
                if volatile_mode:
                    value = value.volatile()
                out[key] = value
            yield out

            if num_batches > 0 and batch_index > num_batches:
                break

    return inner_func

raw_data = RawSurnames().get_data()

raw_data.head()

vectorizer = SurnamesVectorizer.fit(raw_data)

vectorizer.nationality_vocab, vectorizer.surname_vocab

vec_train = vectorizer.transform(raw_data, split='train')

vec_train.x_surnames, vec_train.x_surnames.shape

vec_train.y_nationalities, vec_train.y_nationalities.shape

# let's say we are making a randomized batch. 
n_data = len(vec_train)
indices = np.random.choice(np.arange(n_data), 
                           size=n_data, 
                           replace=False)

batch_indices = indices[:10]
batched_x = vec_train.x_surnames[batch_indices]
batched_x.shape

import torch
from torch import LongTensor
from torch.autograd import Variable

n_surnames = len(vectorizer.surname_vocab)
# padding_idx is very important!
emb = torch.nn.Embedding(embedding_dim=8, num_embeddings=n_surnames, padding_idx=0)

torch_x = Variable(LongTensor(batched_x))
x_seq = emb(torch_x)
x_seq.size()

# where this swaps 1 and 0. if we did it twice, it would swap back. 
x_seq_on_dim0 = x_seq.permute(1, 0, 2)
x_seq_on_dim0.size()

x_5th_step = x_seq_on_dim0[4, :, :]
x_5th_step.size()

