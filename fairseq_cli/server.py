import logging
import math
import os
import sys

import editdistance
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging.meters import StopwatchMeter, TimeMeter
import soundfile
from fairseq.data import Dictionary
from io import BytesIO
import numpy as np
import torch.nn.functional as F

try:
    from gevent import monkey
    monkey.patch_all()
    from flask import Flask, request
    from gevent import pywsgi
    from scipy.signal import resample
except ImportError as err:
    err = str(err) + '\ntry: pip install flask gevent scipy'
    raise err

app = Flask(__name__)


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def do_resample(arr, from_sr, to_sr):
    new_arr, _ = resample(
        arr,
        int(len(arr) * to_sr / from_sr),
        np.arange(len(arr)))
    return new_arr


@app.route("/stt", methods=["POST"])
def stt():
    models = app.config.get("_models")
    generator = app.config.get("_generator")
    target_dict = app.config.get("_target_dict")
    f = request.args.get('format')

    data = request.get_data()
    psudo_file = BytesIO(data)

    if f == 'raw':
        arr, sr = soundfile.read(
            psudo_file,
            samplerate=16000,
            channels=1,
            subtype='PCM_16',
            format='RAW')
    else:
        arr, sr = soundfile.read(psudo_file)
        if sr != 16000:
            arr = do_resample(arr, sr, 16000)

    feats = wav_to_feature(arr)

    sample = dict()
    net_input = dict()

    net_input["source"] = feats.unsqueeze(0)

    padding_mask = torch.BoolTensor(
        net_input["source"].size(1)).fill_(False).unsqueeze(0)

    net_input["padding_mask"] = padding_mask
    sample["net_input"] = net_input

    with torch.no_grad():
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        emissions = generator.get_emissions(models, encoder_input)

        hypo = generator.generate(models, sample, prefix_tokens=None)

    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())

    result = post_process(hyp_pieces, 'letter')

    return result.lower()


@app.route("/health", methods=["GET"])
def health():
    return "1"


@app.route("/config-id", methods=["GET"])
def config_id():
    return app.config.get("_config_id", "")
    
# @app.route("/warmup", methods=["GET"])
# def warmup():
    

def add_asr_argument(parser):
    parser.add_argument("--kspmodel", default=None,
                        help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary\
output units",
    )
    try:
        parser.add_argument(
            "--lm-weight",
            "--lm_weight",
            type=float,
            default=0.2,
            help="weight for lm while interpolating with neural score",
        )
    except:
        pass
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    parser.add_argument(
        "--w2l-decoder",
        choices=["viterbi", "kenlm", "fairseqlm"],
        help="use a w2l decoder",
    )
    parser.add_argument("--lexicon", help="lexicon for w2l decoder")
    parser.add_argument("--unit-lm", action="store_true",
                        help="if using a unit lm")
    parser.add_argument("--kenlm-model", "--lm-model",
                        help="lm model for w2l decoder")
    parser.add_argument("--beam-threshold", type=float, default=25.0)
    parser.add_argument("--beam-size-token", type=float, default=100)
    parser.add_argument("--word-score", type=float, default=1.0)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument(
        "--dump-emissions",
        type=str,
        default=None,
        help="if present, dumps emissions into this file and exits",
    )
    parser.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="if present, dumps features into this file and exits",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )
    parser.add_argument(
        "--dict-path",
        type=str,
        default=None,
        help="dict path",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=5000,
        help="fairseq server port",
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default="",
        help="config ID",
    )
    return parser


def check_args(args):
    # assert args.path is not None, "--path required for generation!"
    # assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def load_models_and_criterions(
    args, arg_overrides=None, task=None, model_state=None
):
    models = []
    criterions = []
    filenames = args.path

    if arg_overrides is None:
        arg_overrides = {}

    arg_overrides["wer_args"] = None
    arg_overrides["_name"] = "serving"

    if filenames is None:
        assert model_state is not None
        filenames = [0]
    else:
        filenames = filenames.split(":")

    for filename in filenames:
        if model_state is None:
            if not os.path.exists(filename):
                raise IOError("Model file not found: {}".format(filename))
            state = checkpoint_utils.load_checkpoint_to_cpu(
                filename, arg_overrides)
        else:
            state = model_state

        if "cfg" in state:
            cfg = state["cfg"]
        else:
            cfg = convert_namespace_to_omegaconf(state["args"])

        if task is None:
            # if hasattr(cfg.task, 'data'):
            #     cfg.task.data = data_path
            # cfg.task.dictionary = args.dictionary
            cfg.task.task = "serving"
            cfg.task.dictionary = args.dictionary

            task = tasks.setup_task(cfg.task)

        model = task.build_model(cfg.model)
        model.load_state_dict(state["model"], strict=True)
        models.append(model)

        criterion = task.build_criterion(cfg.criterion)
        if "criterion" in state:
            criterion.load_state_dict(state["criterion"], strict=True)
        criterions.append(criterion)
    return models, criterions, task


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


def build_generator(args, task):
    w2l_decoder = getattr(args, "w2l_decoder", None)
    if w2l_decoder == "viterbi":
        from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

        return W2lViterbiDecoder(args, task.target_dictionary)
    elif w2l_decoder == "kenlm":
        from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

        return W2lKenLMDecoder(args, task.target_dictionary)
    elif w2l_decoder == "fairseqlm":
        from examples.speech_recognition.w2l_decoder import W2lFairseqLMDecoder

        return W2lFairseqLMDecoder(args, task.target_dictionary)
    else:
        print(
            "only wav2letter decoders with (viterbi, kenlm, fairseqlm) options are supported at the moment"
        )


def process_predictions(
    args, tgt_dict, target_tokens
):

    tgt_pieces = tgt_dict.string(target_tokens)
    return post_process(tgt_pieces, args.post_process)
    # for hypo in hypos[: min(len(hypos), args.nbest)]:
    #     hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())

    #     if "words" in hypo:
    #         hyp_words = " ".join(hypo["words"])
    #     else:
    #         hyp_words = post_process(hyp_pieces, args.post_process)

    #     if res_files is not None:
    #         print(
    #             "{} ({}-{})".format(hyp_pieces, speaker, id),
    #             file=res_files["hypo.units"],
    #         )
    #         print(
    #             "{} ({}-{})".format(hyp_words, speaker, id),
    #             file=res_files["hypo.words"],
    #         )

    #     tgt_pieces = tgt_dict.string(target_tokens)
    #     tgt_words = post_process(tgt_pieces, args.post_process)

    #     if res_files is not None:
    #         print(
    #             "{} ({}-{})".format(tgt_pieces, speaker, id),
    #             file=res_files["ref.units"],
    #         )
    #         print(
    #             "{} ({}-{})".format(tgt_words, speaker, id), file=res_files["ref.words"]
    #         )
    #         # only score top hypothesis
    #         if not args.quiet:
    #             logger.debug("HYPO:" + hyp_words)
    #             logger.debug("TARGET:" + tgt_words)
    #             logger.debug("___________________")

    #     hyp_words = hyp_words.split()
    #     tgt_words = tgt_words.split()
    #     return editdistance.eval(hyp_words, tgt_words), len(tgt_words)


class ExistingEmissionsDecoder(object):
    def __init__(self, decoder, emissions):
        self.decoder = decoder
        self.emissions = emissions

    def generate(self, models, sample, **unused):
        ids = sample["id"].cpu().numpy()
        try:
            emissions = np.stack(self.emissions[ids])
        except:
            print([x.shape for x in self.emissions[ids]])
            raise Exception("invalid sizes")
        emissions = torch.from_numpy(emissions)
        return self.decoder.decode(emissions)


def wav_to_feature(wav):
    def postprocess(feats):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats
    feats = torch.from_numpy(wav).float()
    feats = postprocess(feats)
    return feats


def main(args):
    check_args(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 4000000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    logger.info("| loading model(s) from {}".format(args.path))
    models, criterions, task = load_models_and_criterions(
        args,
        arg_overrides=eval(args.model_overrides),  # noqa
        task=None,
        model_state=None,
    )
    optimize_models(args, use_cuda, models)

    model = models[0]
    model.eval()

    logger.info(args)
    generator = build_generator(args, task)
    app.config["_models"] = models
    app.config["_generator"] = generator

    target_dict = Dictionary.load(args.dictionary)
    app.config["_target_dict"] = target_dict
    app.config["_config_id"] = args.config_id

    # hypos = task.inference_step(generator, models, sample, prefix_tokens)

    # sample = dict()
    # net_input = dict()

    server = pywsgi.WSGIServer(('0.0.0.0', args.server_port), app)
    server.serve_forever()


def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_argument(parser)
    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)

    args.criterion = "ctc"
    main(args)


if __name__ == "__main__":
    cli_main()
