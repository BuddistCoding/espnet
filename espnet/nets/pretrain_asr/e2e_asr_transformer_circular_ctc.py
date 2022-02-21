# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
import logging
import math

import numpy
import torch

import chainer
from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

from espnet.nets.pytorch_backend.transformer.encoder_list import Encoder as Encoder_list
from torch.nn import CrossEntropyLoss, MSELoss

import time

class Cir_Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss, phn_loss, phn_loss_2, loss_ce, loss_mse):
        """Report at every step."""
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"pinyin_ctc": phn_loss}, self)
        reporter.report({"pinyin_ctc after text encoder": phn_loss_2}, self)
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer_ctc": cer_ctc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)
        reporter.report({"Cross Entropy loss": loss_ce}, self)
        reporter.report({"MSE loss between 8th & 12th": loss_mse }, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.intermediate_ctc_weight = args.intermediate_ctc_weight
        self.intermediate_ctc_layers = []
        if args.intermediate_ctc_layer != "":
            self.intermediate_ctc_layers = [
                int(i) for i in args.intermediate_ctc_layer.split(",")
            ]

        self.elayers = args.elayers
        self.text_elayers = args.text_elayers
        self.phoneme_input_layer = args.phoneme_input_layer
        self.phoneme_output_layer = args.phoneme_output_layer

        if self.phoneme_input_layer > 0 and self.phoneme_output_layer > 0:
            assert(self.phoneme_input_layer <= self.phoneme_output_layer)
        self.add_text_encoder = args.add_text_encoder

        self.mode = args.train_mode if args.train_mode == 'text' else 'asr'

        self.ce = CrossEntropyLoss(ignore_index = -1)
        self.mse = MSELoss()

        if args.phn_ctc_weight > 0.0:
            
            self.text_embed = torch.nn.Sequential(
                torch.nn.Embedding(
                    odim, 
                    args.adim,
                    padding_idx=0,
                ),
                PositionalEncoding(args.adim, 0.1)
            )

            logging.warning(f'asr elayers:{args.elayers}')
            logging.warning(f'text_elayers:{args.text_elayers}')

            self.encoder = Encoder_list(
                idim=idim,
                selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_encoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.eunits,
                num_blocks=(args.elayers + args.text_elayers),
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                stochastic_depth_rate=args.stochastic_depth_rate,
                intermediate_layers=self.intermediate_ctc_layers,
            )
            
            self.phn_ctc = CTC(
                args.pdim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )


        else:
            logging.warning('args.phn_ctc_weight == 0.0, so No Phn_CTC')
            logging.warning(f'elayers: {self.elayers}')
            self.encoder = Encoder(
                idim=idim,
                selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_encoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                stochastic_depth_rate=args.stochastic_depth_rate,
                intermediate_layers=self.intermediate_ctc_layers,
            )
            self.phn_ctc = None
        

        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_decoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
            self.criterion = LabelSmoothingLoss(
                odim,
                ignore_id,
                args.lsm_weight,
                args.transformer_length_normalized_loss,
            )
        else:
            self.decoder = None
            self.criterion = None

        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Cir_Reporter()

        self.reset_parameters(args)
        self.adim = args.adim  # used for CTC (equal to d_model)
        self.mtlalpha = args.mtlalpha
        self.phn_ctc_weight = args.phn_ctc_weight
        self.ce_weight = args.ce_weight
        self.forward_again = args.forward_again

        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None
        
        self.use_hyp = args.use_hyp if (args.use_hyp == "hyp" or args.use_hyp == "tru") else "hid"

        
        logging.warning(f'add_text_encoder:{self.add_text_encoder}')
        logging.warning(f'use_hyp:{self.use_hyp}')
        logging.warning(f'mode:{self.mode}')
        logging.warning(f'forward again:{self.forward_again}')
        logging.warning(f'phoneme_input_layer:{self.phoneme_input_layer}')
        logging.warning(f'phoneme_output_layer:{self.phoneme_output_layer}')
        logging.warning(f'phoneme_ctc_weight:{self.phn_ctc_weight}')
        logging.warning(f'Cross Entropy weight:{self.ce_weight}')
        logging.warning(f'encoders: \n{self.encoder}')

        if (self.mode == 'asr' and self.text_elayers > 0):
            for i , layer in enumerate(self.encoder.encoders[self.elayers: self.elayers + self.text_elayers]):
                logging.warning(f'freeze layer {self.elayers + i + 1}')
                for param in layer.parameters():
                    param.requires_grad = False
        

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        ys_phn = None
        if len(ys_pad) == 1:
            ys_pad = ys_pad[0]
        elif len (ys_pad) == 2:
            ys_pad, ys_phn = ys_pad
        else:
            logging.warning(f'ys_pad:{ys_pad}')
            assert (len(ys_pad) <= 2)

        olens = None
        if len(ilens) == 1:
            ilens = ilens[0]
        elif len(ilens) == 2:
            ilens, olens = ilens
        else:
            logging.warning(f'ilens:{ilens}')
            assert (len(ilens) <= 2)
        
        if self.mode == 'text':
            xs_pad = ys_pad.clone()
            ilens = olens.clone().tolist()

        else:
            ilens = ilens.tolist()  
        
        
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens).to(xs_pad.device).unsqueeze(-2)
        # logging.warning(f'First input 8-layer encoder')
        if (self.mode == 'asr'):
            hs_pad, hs_mask = self.encoder.embed(xs_pad, src_mask)
        else: # Text mode
            xs_pad[xs_pad == -1] = 0
            hs_pad = self.text_embed(xs_pad)
            hs_mask = src_mask

        hs_phn_1 = None
        hs_phn_2 = None
        hs_intermediates = []
        
        # logging.warning(f'hs_pad after embed:{hs_pad.shape}')

        if (self.mode == 'asr'):
            for i, layer in enumerate(self.encoder.encoders[:self.elayers]):
                hs_pad, hs_mask = layer(hs_pad, hs_mask)
                if (i + 1 == self.phoneme_output_layer):
                    if (self.encoder.normalize_before):
                        hs_phn_1 = self.encoder.after_norm(hs_pad)
                    else:
                        hs_phn_1 = hs_pad.clone()
        else :
            for i, layer in enumerate(self.encoder.encoders[self.elayers:]):
                hs_pad, hs_mask = layer(hs_pad, hs_mask)

            for i, layer in enumerate(self.encoder.encoders[self.phoneme_input_layer - 1 : self.elayers]):
                hs_pad, hs_mask = layer(hs_pad, hs_mask)
                if (i + 1 == self.phoneme_output_layer):
                    if (self.encoder.normalize_before):
                        hs_phn_1 = self.encoder.after_norm(hs_pad)
                    else:
                        hs_phn_1 = hs_pad.clone()

        hs_pad_2 = None
        if (self.add_text_encoder):
            hs_pad_2 = hs_pad.clone()

        if (self.encoder.normalize_before):
            hs_pad = self.encoder.after_norm(hs_pad)
        
        self.hs_pad = hs_pad.clone()

        if (self.mode == 'asr'):
            # FOR Phoneme CTC
            if (self.add_text_encoder):
                ys_text = ys_pad.clone()
                if (self.use_hyp == "hid"):
                    for i in range (self.elayers, self.elayers + self.text_elayers):
                        hs_pad_2, hs_mask= self.encoder.encoders[i](hs_pad_2, hs_mask)
                    
                    # elif (self.use_hyp == "tru"):
                    #     hs_pad_2 = self.ctc.ctc_lo.weight[ys_text][:]

                    #     y_len = []
                    #     for y in ys_text:
                    #         y_len.append(torch.count_nonzero(y != -1))

                    #     hs_mask = make_non_pad_mask(y_len).to(hs_pad_2.device).unsqueeze(-2)

                    #     for i in range (self.elayers, self.elayers + self.text_elayers):
                    #         hs_pad_2, hs_mask= self.encoder.encoders[i](hs_pad_2, hs_mask)

                else: # use hypothesis from CTC
                    pass 
            
                # logging.warning(f'hs_pad after text_encoder:{hs_pad}')
                
                if (self.forward_again):
                    if (self.phoneme_input_layer > 0):
                        for i , layer in (self.encoder.encoders[self.phoneme_input_layer - 1 : self.elayers]):
                            hs_pad_2 = layer(hs_pad_2, hs_mask)
                            if (i + self.phoneme_input_layer == self.phoneme_output_layer):
                                if (self.encoder.normalize_before):
                                    hs_phn_2 = self.encoder.after_norm(hs_phn_2)
                                else:
                                    hs_phn_2 = hs_pad_2.clone()

                        if (self.encoder.normalize_before):
                            hs_pad_2 = self.encoder.after_norm(hs_pad_2)
                else: # Calculate the MSE between 12th output and 8th layer
                    pass
    
        # 2. forward decoder
        if self.decoder is not None:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        loss_intermediate_ctc = 0.0
        loss_phn_ctc = 0.0
        loss_phn_ctc_2 = 0.0
        loss_ce = 0.0
        loss_mse = 0.0
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)

            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            loss_ctc_2 = 0.0
            if (hs_pad_2 is not None and self.forward_again):
                loss_ctc_2 = self.ctc(hs_pad_2.view(batch_size, -1, self.adim), hs_len, ys_pad)

            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

            if self.intermediate_ctc_weight > 0 and self.intermediate_ctc_layers:
                for hs_intermediate in hs_intermediates:
                    # assuming hs_intermediates and hs_pad has same length / padding
                    loss_inter = self.ctc(
                        hs_intermediate.view(batch_size, -1, self.adim), hs_len, ys_pad
                    )
                    loss_intermediate_ctc += loss_inter

                loss_intermediate_ctc /= len(self.intermediate_ctc_layers)


            if self.phn_ctc_weight > 0:
                if hs_phn_1 is not None:
                    loss_phn_ctc = self.phn_ctc(hs_phn_1.view(batch_size, -1, self.adim), hs_len, ys_phn)
                    # loss_phn_ctc += loss_phn

                if hs_phn_2 is not None:
                    # xs_pad,  hs_mask, _, hs_phn = self.encoder(xs_pad, src_mask, h = hs_pad)
                    loss_phn_ctc_2 = self.phn_ctc(hs_phn_2.view(batch_size, -1, self.adim), hs_len, ys_phn)

        if (self.mode == 'text' and self.ce_weight > 0):
            loss_ce = self.ce(self.ctc.softmax(self.ctc.dropout(hs_pad)).view(batch_size, self.odim, -1), ys_pad)
        
        if (self.mode == 'asr' and self.add_text_encoder and not self.forward_again):
            loss_mse = self.mse(hs_pad, hs_pad_2)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    1 - self.intermediate_ctc_weight
                ) * loss_ctc + self.intermediate_ctc_weight * loss_intermediate_ctc

            if self.phn_ctc_weight > 0:
                if (loss_ctc_2 > 0):
                    loss_ctc = loss_ctc + loss_ctc_2

                if (loss_phn_ctc_2 > 0):
                    loss_phn_ctc = loss_phn_ctc + loss_phn_ctc_2
                
                loss_ctc_data = (1 - self.phn_ctc_weight) * loss_ctc + self.phn_ctc_weight * loss_phn_ctc

                self.loss = (1 - self.ce_weight) * loss_ctc_data

                if (self.ce_weight > 0):
                    self.loss += self.ce_weight * loss_ce
                
                if (loss_mse > 0):
                    self.loss += loss_mse
            
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                        (1 - alpha - self.phn_ctc_weight) * loss_att
                        + alpha * loss_ctc
                        + self.intermediate_ctc_weight * loss_intermediate_ctc
                    )
                
            elif self.phn_ctc_weight > 0:
                self.loss = (
                    (1 - self.ce_weight) * (
                        (1 - alpha) * loss_att
                        + alpha * loss_ctc
                        + self.phn_ctc_weight * (loss_phn_ctc + loss_phn_ctc_2)
                    )
                        + self.ce_weight * loss_ce
                    )
                    
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)
            

        loss_data = float(self.loss)
        # logging.warning(f'\nctc_loss:{loss_ctc} \nphn_ctc_loss:{loss_phn_ctc} \nloss_data:{loss_data}')
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, 
                loss_att_data, 
                self.acc, 
                cer_ctc,
                cer, 
                wer, 
                loss_data, 
                loss_phn_ctc, 
                loss_phn_ctc_2, 
                loss_ce,
                loss_mse,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        
        # logging.warning(f'end time:{time.time() - start_time}')
        # logging.warning(f'total time:{time.time() - org_time}')

        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        if (self.mode == 'asr'):
            x, _ = self.encoder.embed(x, None)
            for i in range(self.elayers):
                # logging.warning(f'Decoding : forward {i + 1} layer.')
                x, _ = self.encoder.encoders[i](x, None)
            
        elif (self.mode == 'text'):
            x, _ = self.encoder.text_embed(x, None)
            for i in range(self.text_elayers):
                x, self.encoder.encoders[self.elayers + i](x, None)
            for i in range(self.phoneme_input_layer - 1, self.elayers):
                # logging.warning(f'layer:{i}')
                hs_pad, hs_mask = self.encoder.encoders[i](hs_pad, hs_mask)   
        
        return x.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # if (self.mode == 'asr'):
        #     x = x[0]
        # elif (self.mode == 'text'):
        #     x = x[1]

        enc_output = self.encode(x).unsqueeze(0)
        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc.argmax(enc_output)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps

        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                prob = ""
                if (isinstance(m, MultiHeadedAttention)):
                    prob = "MultiheadAtt"
                elif (isinstance(m, DynamicConvolution)):
                    prob = "DynamicConv"
                elif (isinstance(m, RelPositionMultiHeadedAttention)):
                    prob = "RelPositionMultiHeadedAtt"
                if (m.attn is not None):
                    ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
