# 0714 수정
# coding=utf-8

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import GOLDEN_TRUTH


class Inputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features

        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader

        # valid
        self.valid_dataloader = DynamicBatchingLoader

        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(
            self,
            input_ids,
            decoder_input_ids, labels,
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)  # 160

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)  # 43
        self.labels = labels

        self.input_len = self.input_length + self.decoder_input_length  # 160+43=203


def featurize(
        bos, eos,
        context, max_input_length,
        response, max_decoder_input_length, emotion_id, intensity_id, strat_id,
):
    context = [c + [eos] for c in context]
    input_ids = sum(context, [])[:-1]
    # (max_input_length) = (160)
    input_ids = input_ids[-max_input_length:]

    # (max_decoder_input_length+3) = (43)
    labels = ([emotion_id] + [intensity_id] + [strat_id] + response + [eos])[:max_decoder_input_length + 3]
    # (max_decoder_input_length+3) = (43)
    decoder_input_ids = [bos] + labels[:-1]

    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

    return InputFeatures(
        input_ids,
        decoder_input_ids, labels,
    )


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    dialog = data['dialog']
    inputs = []
    context = []

    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        text = process(text)

        if dialog[i]['speaker'] == 'sys':
            emotion_id = process('[' + dialog[i]['emotion'] + ']')  # 54944~54955
            assert len(emotion_id) == 1
            emotion_id = emotion_id[0]

            intensity_id = process('[' + dialog[i]['intensity'] + ']')  # 54956~54961
            assert len(intensity_id) == 1
            intensity_id = intensity_id[0]

            strat_id = process('[' + dialog[i]['strategy'] + ']')  # 54962~54969
            assert len(strat_id) == 1
            strat_id = strat_id[0]

            

        if i > 0 and dialog[i]['speaker'] == 'sys':
            res = {
                'context': context.copy(),
                'response': text,
                'emotion_id': emotion_id,
                'intensity_id': intensity_id,
                'strat_id': strat_id,
            }

            inputs.append(res)

        if dialog[i]['speaker'] == 'sys':
            text = [emotion_id] + [intensity_id] + [strat_id] + text

        context = context + [text]

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos,
            ipt['context'], max_input_length,
            ipt['response'], max_decoder_input_length, ipt['emotion_id'], ipt['intensity_id'], ipt['strat_id'],
        )  # InputFeatures object return
        features.append(feat)
    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
            assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id
            assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

        # (batch_size, max_input_length) = (16, 160)
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                                 batch_first=True, padding_value=pad)
        # (batch_size, max_input_length) = (16, 160)
        attention_mask = pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                                      batch_first=True, padding_value=0.)
        input_length = torch.tensor([f.input_length for f in features], dtype=torch.long)

        if not infer:
            # (batch_size, max_decoder_input_length+3) = (16, 43)
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                                             batch_first=True, padding_value=pad)
            # (batch_size, max_decoder_input_length+3) = (16, 43)
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            labels = None

        # (batch_size) = (16) 0~11
        emotion_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(toker) + 26
        # (batch_size) = (16) 0~7 +14 => 8~13 / +8 => 0~7
        intensity_id = torch.tensor([f.labels[1] for f in features], dtype=torch.long) - len(toker) + 14
        # (batch_size) = (16) +26 => 13~25 / +14 => 8~13 / +8 => 0~7
        strat_id = torch.tensor([f.labels[2] for f in features], dtype=torch.long) - len(toker) + 6

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'input_length': input_length,

            'decoder_input_ids': decoder_input_ids,
            'labels': labels,

            'emotion_id': emotion_id,
            'intensity_id': intensity_id,
            'strat_id': strat_id,

        }

        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """

    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()

            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []

            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch

        except StopIteration:
            pass

    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


# for inference
def prepare_infer_batch(features, toker, interact=None):
    res = FeatureDataset.collate(features, toker, True)

    res['batch_size'] = res['input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_emotion_id': 'pred_emotion_id',
        'cls_intensity_id': 'pred_intensity_id',
        'cls_strat_id': 'pred_strat_id',
    }

    if interact is None and GOLDEN_TRUTH:
        other_res['cls_emotion_id'] = res.get('emotion_id')
        other_res['cls_intensity_id'] = res.get('intensity_id')
        other_res['cls_strat_id'] = res.get('strat_id')
    else:
        other_res['cls_emotion_id'] = res.pop('emotion_id')
        other_res['cls_intensity_id'] = res.pop('intensity_id')
        other_res['cls_strat_id'] = res.pop('strat_id')

    return res


def get_infer_batch(infer_input_file, toker, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()

    features = []
    sample_ids = []
    posts = []
    references = []
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(toker.decode(ipt['context'][-1]))
            references.append(toker.decode(ipt['response']))
            sample_ids.append(sample_id)

            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids
