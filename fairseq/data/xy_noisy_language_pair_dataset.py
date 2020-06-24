# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#


import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    src_target = merge('source_output', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    src_target = src_target.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        ntokens = sum(len(s['target']) for s in samples)
        target = merge('output', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=False,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'src_target': src_target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def generate_dummy_batch(num_tokens, collate_fn, src_vocab, tgt_vocab, src_len=128, tgt_len=128):
    """Return a dummy batch with a given number of tokens."""
    bsz = num_tokens // max(src_len, tgt_len)
    return collate_fn([
        {
            'id': i,
            'source': src_vocab.dummy_sentence(src_len),
            'target': tgt_vocab.dummy_sentence(tgt_len),
            'output': tgt_vocab.dummy_sentence(tgt_len),
            'source_output': src_vocab.dummy_sentence(src_len),
        }
        for i in range(bsz)
    ])


class XYNoisyLanguagePairDataset(FairseqDataset):
    """
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
                        |
                        V
    [x1, x2, x3, x4, x5] [y1, _, y3, _, y5]
    source: [x1, _, x3, _, x5]
    prev_output_tokens: [y1, _, y3, _, y5]
    output: [PAD, y2, PAD, y4, PAD]
    """
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        mask_source_rate=0.15, mask_span=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        self.src_vocab = src_dict
        self.tgt_vocab = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.mask_source_rate = mask_source_rate
        self.mask_span = mask_span

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]
        
        tgt_list = tgt_item.tolist()
        src_list = src_item.tolist()

        target = tgt_list.copy()
        output = tgt_list.copy()
        source = src_list.copy()
        source_output = src_list.copy()

        source_len_mask = max(1, round(len(src_list) * self.mask_source_rate))
        start = self.mask_start(len(src_list)-source_len_mask)
        source_id_mask = np.arange(start, start+source_len_mask+1)

        target_num_mask = np.random.randint(1, len(tgt_list) + 1)
        
        span_target_id_mask = self.masking_span(target_num_mask, len(tgt_list), is_target=True)
        
        random_target_id_mask = np.arange(len(tgt_list))
        np.random.shuffle(random_target_id_mask)
        random_target_id_mask = sorted(random_target_id_mask[:target_num_mask])

        target_id_mask = random_target_id_mask
        if self.mask_span:
            target_id_mask = span_target_id_mask

        for i, w in enumerate(tgt_list):
            if i in target_id_mask:
                target[i] = self.mask_word(w)
            else:
                output[i] = self.tgt_vocab.pad()

        for i, w in enumerate(src_list):
            if isinstance(source_id_mask, dict):
                if i in source_id_mask.keys():
                    source[i] = self.mask_word(w, source=True, p=source_id_mask[i])
                else:
                    source_output[i] = self.src_vocab.pad()
            else:
                if i in source_id_mask:
                    source[i] = self.mask_word(w, source=True)
                else:
                    source_output[i] = self.src_vocab.pad()

        assert len(target) == len(output)
            
        return {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': torch.LongTensor(output),
            'source_output': torch.LongTensor(source_output)
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_vocab.pad(), eos_idx=self.src_vocab.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        return generate_dummy_batch(num_tokens, self.collater, self.src_vocab, self.tgt_vocab, src_len, tgt_len)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def mask_word(self, w, source=False, p=None):
        if source:
            voc = self.src_vocab
        else:
            voc = self.tgt_vocab

        p = np.random.random() if p is None else p
        if p >= 0.2:
            return voc.mask_index
        elif p >= 0.1:
            return np.random.randint(voc.nspecial, len(voc))
        else:
            return w

    def masking_span(self, num_mask, total_len, is_target=False):
        span_lens = []
        temp_num_mask = num_mask
        while True:
            if is_target:
                span_len_one = 2
            else:
                span_len_one = max(1, np.random.geometric(p=0.2))
                while span_len_one>10:
                    span_len_one = max(1, np.random.geometric(p=0.2))

            if temp_num_mask >= span_len_one:
                span_lens.append(span_len_one)
                temp_num_mask -= span_len_one
            else:
                span_lens.append(temp_num_mask)
                break
        assert sum(span_lens) == num_mask

        np.random.shuffle(span_lens)
        num_to_insert = total_len - num_mask
        slot_to_insert = len(span_lens) + 1
        slot_insert_nums = []
        for _ in range(slot_to_insert):
            if num_to_insert == 0:
                slot_insert_nums.append(0)
            else:
                insert_num = np.random.randint(0, num_to_insert)
                slot_insert_nums.append(insert_num)
                num_to_insert -= insert_num
        if num_to_insert != 0:
            slot_insert_nums[-1] += num_to_insert

        slot_id_mask = []
        now_id = 0
        for i in range(len(span_lens)):
            now_id += slot_insert_nums[i]
            end_id = now_id + span_lens[i]
            id_mask = np.arange(now_id, end_id)
            slot_id_mask.append(id_mask)
            now_id = end_id

        id_mask_dict = {}
        for id_mask in slot_id_mask:
            p = np.random.random()
            for id_ in id_mask:
                id_mask_dict[id_] = p
        return id_mask_dict

    def mask_start(self, end):
        p = np.random.random()
        if p >= 0.8:
            return 0
        elif p >= 0.6:
            return end
        else:
            return np.random.randint(end)

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
