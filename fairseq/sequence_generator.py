# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn.functional as F

from fairseq import search, utils, checkpoint_utils
from fairseq.models import FairseqIncrementalDecoder


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y


def assign_multi_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y.view(-1)[i.view(-1).nonzero()]


def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y


def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]


class SequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        stop_early=True,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
        mask_pred_iter=10,
        use_golden_length=False,
        args=None,
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            stop_early (bool, optional): stop generation immediately after we
                finalize beam_size hypotheses, even though longer hypotheses
                might have better normalized scores (default: True)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.mask = tgt_dict.mask()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        self.use_golden_length = use_golden_length
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.mask_pred_iter = mask_pred_iter

        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = search.BeamSearch(tgt_dict)

    @torch.no_grad()
    def generate(
        self,
        models,
        sample,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
        """
        model = EnsembleModel(models)
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size
        use_golden_length = self.use_golden_length

        finalized = [[] for i in range(bsz)]
        
        # non-autoregressive decoding

        def get_hypo_nat(decoded_id):
            return {
                'tokens': decoded_id,
                'score': 0.0,
                'attention': None,  # src_len x tgt_len
                'alignment': None,
                'positional_scores': torch.Tensor([0.0]),
            }

        def copy_batches(tensor, num_copies):
            if tensor is None:
                return None
            x_size = tensor.size()
            tensor = tensor.contiguous().view(x_size[0], 1, -1)
            tensor = tensor.repeat(1, num_copies, 1)
            if len(x_size)==2:
                return tensor.view(-1, x_size[1])
            elif len(x_size)==3:
                return tensor.view(-1, x_size[1], x_size[2])
            else:
                raise NotImplementedError

        def select_worst(token_probs, num_mask):
            bsz, seq_len = token_probs.size()
            masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
            masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
            return torch.stack(masks, dim=0)

        encoder_outs = model.forward_encoder(encoder_input)

        if use_golden_length:
            gold_target_len = sample['target'].ne(self.pad).sum(-1)
            beam_starts = gold_target_len - (beam_size - 1) // 2
            beam_ends = gold_target_len + beam_size // 2 + 1
            beam = torch.stack([torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
        else:
            predicted_lengths  = encoder_outs[0]['predicted_lengths']
            beam = predicted_lengths.topk(beam_size, dim=1)[1]
        beam[beam<2] = 2

        max_len = beam.max().item()
        length_mask = torch.triu(src_tokens.new(max_len, max_len).fill_(1).long(), 1)
        length_mask = torch.stack([length_mask[beam[batch] - 1] for batch in range(bsz)], dim=0)
        tgt_tokens = src_tokens.new(bsz, beam_size, max_len).fill_(self.mask)
        tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * self.pad
        tgt_tokens = tgt_tokens.view(bsz * beam_size, max_len)
        pad_mask = tgt_tokens.eq(self.pad)
        seq_lens = tgt_tokens.size(1) - pad_mask.sum(dim=1)

        encoder_outs_value = copy_batches(encoder_outs[0]['encoder_out'].transpose(0,1), beam_size)
        encoder_outs_value = encoder_outs_value.transpose(0,1)
        encoder_padding = copy_batches(encoder_outs[0]['encoder_padding_mask'], beam_size)

        encoder_outs = [{'encoder_out': encoder_outs_value, 'encoder_padding_mask': encoder_padding, 'predicted_lengths': encoder_outs[0]['predicted_lengths']}]
        
        tgt_tokens, token_probs, _ = model.forward_decoder(
            tgt_tokens, encoder_outs, 
            temperature=self.temperature,
        )
        assign_single_value_byte(tgt_tokens, pad_mask, self.pad)
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        for i in range(1, self.mask_pred_iter+1):
            num_mask = (seq_lens.float()*(1.0-i/self.mask_pred_iter)).long()
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, self.mask)
            assign_single_value_byte(tgt_tokens, pad_mask, self.pad)
            new_tgt_tokens, new_token_probs, all_token_probs = model.forward_decoder(
                tgt_tokens, encoder_outs, 
                temperature=self.temperature,
            )

            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            
            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            assign_single_value_byte(tgt_tokens, pad_mask, self.pad)

        lprobs = token_probs.log().sum(-1)
        hypotheses = tgt_tokens.view(bsz, beam_size, max_len)
        lprobs = lprobs.view(bsz, beam_size)
        tgt_lengths = (1 - length_mask).sum(-1)

        # add len penalty
        length_penalty = ((5.0 + tgt_lengths.float()) ** self.len_penalty
                          / (6.0 ** self.len_penalty))
        length_penalty = length_penalty.view((bsz, beam_size))
        avg_log_prob = lprobs / length_penalty

        best_lengths = avg_log_prob.max(-1)[1]
        hypotheses = torch.stack([hypotheses[b, l, :] for b, l in enumerate(best_lengths)], dim=0)

        for i in range(bsz):
            finalized[i].append(get_hypo_nat(hypotheses[i]))

        return finalized

class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1.,):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, log_probs,
        temperature=1.,
    ):
        decoder_out = list(model.decoder(tokens, encoder_out=encoder_out)
        )
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn['attn']
        if attn is not None:
            if type(attn) is dict:
                attn = attn['attn']

        probs = F.softmax(decoder_out[0], dim=-1)
        max_probs, idx = probs.max(dim=-1)
        return idx, max_probs, probs

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)
