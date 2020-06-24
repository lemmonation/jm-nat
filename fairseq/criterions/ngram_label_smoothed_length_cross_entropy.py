# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch
from torch.autograd import Variable
from collections import Counter


@register_criterion('ngram_label_smoothed_length_cross_entropy')
class NgramLabelSmoothedLengthCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, length_loss, src_nll_loss, ngram_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'length_loss': utils.item(length_loss.data) if reduce else length_loss.data,
            'ngram_loss': utils.item(ngram_loss.data) if reduce else ngram_loss.data,
            'src_nll_loss': utils.item(src_nll_loss.data) if reduce else src_nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        # compute length prediction loss
        length_lprobs = net_output[1]['predicted_lengths']
        length_target = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).sum(-1).unsqueeze(-1)
        length_loss = -length_lprobs.gather(dim=-1, index=length_target)

        sm_probs = model.get_normalized_probs(net_output, log_probs=False)
        sm_probs = sm_probs.view(-1, sm_probs.size(-1))
        sm_probs = sm_probs*non_pad_mask.float()
        ngram_loss = self.get_ngram_loss(target, sm_probs, length_target)

        src_lprobs = utils.log_softmax(net_output[1]['encoder_out'], dim=-1)
        src_lprobs = src_lprobs.view(-1, src_lprobs.size(-1))
        src_target = sample['src_target'].view(-1, 1)
        src_non_pad_mask = src_target.ne(self.padding_idx)
        src_nll_loss = -src_lprobs.gather(dim=-1, index=src_target)[src_non_pad_mask]

        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            length_loss = length_loss.sum()
            src_nll_loss = src_nll_loss.sum()
            ngram_loss = ngram_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss + 0.1*length_loss + 0.01*src_nll_loss + 0.01*ngram_loss
        return loss, nll_loss, length_loss, src_nll_loss, ngram_loss

    def get_ngram_loss(self, target, probs, target_length):
        target_length = target_length.squeeze(-1).long().tolist()
        batch_size = len(target_length)
        targets = target.squeeze(-1).data.tolist()
        targets = self.shape(targets, target_length)

        matchs = []
        gram_match = self.twogram_match(targets, target_length, probs)
        for i in range(batch_size):
            matchs.append(gram_match[i][0]/gram_match[i][1])

        gram_loss = sum(matchs).div(batch_size)

        return gram_loss

    def twogram_match(self, targets, target_lens, probs):
        batch_size = len(target_lens)
        batch_match = []
        end = 0
        for i in range(batch_size):
            begin = end
            end = begin + target_lens[i]
            curr_tar = targets[i]

            if target_lens[i] < 2:
                batch_match.append((0,1e-5))
                continue

            two_grams = Counter()
            for j in range(len(curr_tar) - 1):
                two_grams[(curr_tar[j], curr_tar[j+1])] += 1

            gram_1, gram_2 = [], []
            gram_count = []
            for two_gram in two_grams:
                gram_1.append(int(two_gram[0]))
                gram_2.append(int(two_gram[1]))
                gram_count.append(two_grams[two_gram])

            match_gram_1 = probs[begin:end-1, gram_1]
            match_gram_2 = probs[begin+1:end, gram_2]
            match_gram = match_gram_1 * match_gram_2
            match_gram = torch.sum(match_gram, dim = 0).view(-1,1)

            gram_count = Variable(torch.Tensor(gram_count).cuda(probs.get_device())).view(-1,1)
            match_gram = torch.min(torch.cat([match_gram,gram_count],dim = -1), dim = -1)[0]
            match_gram = torch.sum(match_gram)
            batch_match.append((match_gram, target_lens[i] - 1))

        return batch_match

    def shape(self, targets, target_lens):
        list_targets = []
        begin = 0
        end = 0
        for length in target_lens:
            end += length
            list_targets.append([str(index) for index in targets[begin:end]])
            begin += length
           
        return list_targets

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'length_loss': sum(log.get('length_loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'src_nll_loss': sum(log.get('src_nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ngram_loss': sum(log.get('ngram_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
