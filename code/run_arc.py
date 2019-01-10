"""BERT finetuning runner."""
import logging
import os
import argparse
import random
import collections
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
DEBUG = False


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 swag_id,
                 context_sentence,
                 start_ending,
                 endings,
                 label=None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = endings
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"swag_id: {self.swag_id}",
            f"context_sentence: {self.context_sentence}",
            f"start_ending: {self.start_ending}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
            f"ending_2: {self.endings[2]}",
            f"ending_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_swag_examples(input_file):
    import json
    import string

    with open(input_file, 'r') as f:
        data = json.load(f)

    examples = []
    answer_mapping = {string.ascii_uppercase[i]: i for i in range(4)}

    for (i, passage_id) in enumerate(data.keys()):
        for question_id in range(len(data[passage_id]['questions'])):
            guid = '%s-%s' % (passage_id, str(question_id))
            article = data[passage_id]['article']
            question = data[passage_id]['questions'][question_id]
            choices = data[passage_id]['options'][question_id]
            label = answer_mapping[data[passage_id]['answers'][question_id].upper()]
            examples.append(
                SwagExample(swag_id=guid, start_ending=question, endings=choices, context_sentence=article, label=label))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training, rev=False):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            if rev:
                tokens = ["[CLS]"] + ending_tokens + ["[SEP]"] + context_tokens_choice + ["[SEP]"]
            else:
                tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 0:
            logger.info("*** Example ***")
            logger.info(f"swag_id: {example.swag_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id=example.swag_id,
                choices_features=choices_features,
                label=label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--bert_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The BERT directory, relative to MODEL_FOLDER")

## Environment parameters
parser.add_argument("--ON_AZURE",
                    default=False,
                    help="if training the mode on azure")
parser.add_argument("--ON_PHILLY",
                    default=False,
                    type=bool,
                    help="if training the mode on philly")
parser.add_argument("--DATA_FOLDER",
                    default=None,
                    type=str,
                    help="The global data folder, for AZURE")
parser.add_argument("--task_name",
                    default=None,
                    type=str,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default='./outputs',
                    type=str,
                    help="The output directory where the model checkpoints will be written.")

## Other parameters
parser.add_argument("--rev",
                    default=False,
                    type=bool,
                    help="Reverse the order of LM or not, default is False")
parser.add_argument("--init_checkpoint",
                    default=None,
                    type=str,
                    help="Initial checkpoint (usually from a pre-trained BERT model), relative to MODEL_FOLDER")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    default=False,
                    type=bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=False,
                    type=bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_lower_case",
                    default=True,
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',
                    type = bool,
                    default=False,
                    #action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--test_feature_file',
                    type=str,
                    default=None,
                    help="test feature file for evaluation, only used when do_eval is True")


args = parser.parse_args()

if args.ON_PHILLY:
    logger.info('Training on PHILLY')
    #BLANK because this is Microsoft Only
elif args.ON_AZURE:
    logger.info('Training on AZURE')
    #BLANK because this is Microsoft Only
else:
    logger.info('Training on LOCAL')

assert args.DATA_FOLDER is not None, "DATA FOLDER cannot be None"

task_name = args.task_name.lower()
args.data_dir = '/'.join([args.DATA_FOLDER, 'data', task_name])  # os.path.join(args.DATA_FOLDER, 'data', task_name)
args.MODEL_FOLDER = '/'.join([args.DATA_FOLDER, 'model'])  # os.path.join(args.DATA_FOLDER, 'model')
args.bert_model = '/'.join([args.MODEL_FOLDER, args.bert_dir])  # os.path.join(args.MODEL_FOLDER, args.bert_dir)
if args.init_checkpoint is not None:
    # args.init_checkpoint = os.path.join(args.MODEL_FOLDER, args.init_checkpoint)
    args.init_checkpoint = '/'.join([args.MODEL_FOLDER, args.init_checkpoint])

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))


##########################################################################
#  Get Machine Configuration and check input arguments
##########################################################################
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))

if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                        args.gradient_accumulation_steps))

args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
logger.info('training batch size on each GPU = %d' % args.train_batch_size)

args.seed = random.randint(0, 100000000)
logger.info('new seed = %d' % args.seed)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

if not args.do_train and not args.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

# if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
#     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
os.makedirs(args.output_dir, exist_ok=True)


##########################################################################
# Prepare for Model
##########################################################################
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
logger.info('loading pretrained bert model from %s' % args.init_checkpoint)
model = BertForMultipleChoice.from_pretrained(args.bert_model, num_choices=4)
if args.init_checkpoint is not None:
    logger.info('loading finetuned model from %s' % args.init_checkpoint)
    model_state_dict = torch.load(args.init_checkpoint)
    model_state_dict_new = collections.OrderedDict()
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        model_state_dict_new[new_key] = model_state_dict[t]
    model.load_state_dict(model_state_dict_new)

if args.fp16:
    model.half()
model.to(device)
if args.local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)


##########################################################################
#  Process Data
##########################################################################
if args.do_train:
    #train_examples = read_swag_examples(os.path.join(args.data_dir, 'train.' + task_name + '.json'))
    train_examples = read_swag_examples('/'.join([args.data_dir, 'train.' + task_name + '.json']))
    num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynaimc_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0

    # Prepare Tensor Data and train Dataloader
    train_features = convert_examples_to_features(
        train_examples, tokenizer, args.max_seq_length, True, args.rev)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # Prepare Tensor Data and dev Dataloader
    # eval_examples = read_swag_examples(os.path.join(args.data_dir, 'dev.'   + task_name + '.json'))
    eval_examples = read_swag_examples('/'.join([args.data_dir, 'dev.'   + task_name + '.json']))
    eval_features = convert_examples_to_features(
        eval_examples, tokenizer, args.max_seq_length, True, args.rev)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids_eval = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask_eval = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids_eval = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_label_eval = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids_eval, all_input_mask_eval, all_segment_ids_eval, all_label_eval)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Training
    model.train()
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", miniters=int(len(train_dataloader)/100))):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                # rescale loss for fp16 training
                # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        # Save a trained model for each epoch
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        #output_model_file = os.path.join(args.output_dir, "%s-%d-%.6f.bin" % (task_name, epoch, args.learning_rate))
        output_model_file = '/'.join([args.output_dir, "%s-%d-%.6f-%d.bin" % (task_name, epoch, args.learning_rate,
                                                                              args.train_batch_size)])
        torch.save(model_to_save.state_dict(), output_model_file)

        # Evaluation on eval dataset
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss/nb_tr_steps}

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

if args.do_eval:
    test_files = args.test_feature_file
    if test_files is None:
        #test_files = os.path.join(args.data_dir, 'test.' + task_name + '.json')
        test_files = '/'.join([args.data_dir, 'test.' + task_name + '.json'])
    for f in test_files.split(','):
        logger.info('for test file %s' % f)
        f = os.path.join(args.DATA_FOLDER, 'data', f)
        test_examples = read_swag_examples(f)

        test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length, True, args.rev)
        logger.info("***** Running Testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        pred_logits = []
        test_labels = []
        for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            pred_logits.append(logits)
            test_labels.append(label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        test_label_flat = np.concatenate(test_labels)
        pred_logits_flat = np.concatenate(pred_logits)

        def np_softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        test_ids = [test_examples[i].swag_id for i in range(0, len(test_examples))]
        answers = ['A', 'B', 'C', 'D']
        probs = np_softmax(pred_logits_flat.T).T
        preds = probs.argmax(1)

        test_res = pd.DataFrame()
        test_res['id'] = test_ids[:len(preds)]
        test_res['answer'] = [answers[i] for i in preds]
        test_res['label'] = [answers[i] for i in test_label_flat]
        for idx_a, a in enumerate(answers):
            test_res['probs' + a] = probs[:, idx_a]
        test_res_file = os.path.basename(args.init_checkpoint) + '-' + os.path.basename(f) + '.csv'
        #test_res.to_csv(os.path.join(args.output_dir, test_res_file))
        test_res.to_csv('/'.join([args.output_dir, test_res_file]))
