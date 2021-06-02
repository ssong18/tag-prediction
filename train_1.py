import argparse
import os
import random
import pickle
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from common.args import *
from common.dataset import TagPredictionTrainDataset, TagPredictionEvalDataset
from common.dataset import remove_answerers_from_question, remove_questions_from_answerer
from common.trainer_1 import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_type', choices=['mathoverflow', 'stackoverflow'], default='mathoverflow')
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for dataloader")
    parser.add_argument('--gpu', type=int, default=-1, help='index of the gpu device to use (cpu if -1)')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999))
    parser.add_argument('--user', type=str, default='hj', choices=['hj', 'bk', 'sy'])
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--evaluate_every', type=int, default=1)
    # model
    parser.add_argument('--negative_samples', type=int, default=64)
    parser.add_argument('--layer_type', choices=['rFF', 'rSA'], default='rFF')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_known_args()[0]

    #################
    ## Environment ##
    #################
    print('[INITIALIZE ENVIRONMENTAL CONFIGS]')
    wandb.login()
    wandb.init(project='tag_prediction')
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(args.gpu))
    device = torch.device('cuda' if args.gpu != -1 else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    if args.data_type == 'mathoverflow':
        data_path = args.data_dir + '/mathoverflow.pickle'
        parser = add_math_overflow_arguments(parser)

    elif args.data_type == 'stackoverflow':
        data_path = args.data_dir + '/stackoverflow.pickle'
        parser = add_stack_overflow_arguments(parser)

    else:
        raise NotImplementedError

    args = parser.parse_args()
    wandb.config.update(args)
    print(args)

    ################
    ## DataLoader ##
    ################
    # Initialize raw-data & loader
    print('[LOAD DATA]')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    question_info_dict = data['question_info_dict']
    answerer_info_dict = data['answerer_info_dict']
    new_question_info_dict = remove_answerers_from_question(question_info_dict,
                                                            answerer_info_dict,
                                                            args.max_answerers,
                                                            args.answerer_removal_criterion)
    new_answerer_info_dict = remove_questions_from_answerer(answerer_info_dict,
                                                            question_info_dict,
                                                            args.max_questions,
                                                            args.question_removal_criterion)
    question_info_dict = new_question_info_dict
    answerer_info_dict = new_answerer_info_dict
    question_to_tags = data['question_to_tags']
    tag_to_idx = data['tag_to_idx']
    train_data = TagPredictionTrainDataset(args,
                                           data['train'],
                                           question_info_dict,
                                           answerer_info_dict,
                                           question_to_tags,
                                           tag_to_idx)
    val_data = TagPredictionEvalDataset(args,
                                        data['val'],
                                        question_info_dict,
                                        answerer_info_dict,
                                        question_to_tags,
                                        tag_to_idx)
    test_data = TagPredictionEvalDataset(args,
                                         data['test'],
                                         question_info_dict,
                                         answerer_info_dict,
                                         question_to_tags,
                                         tag_to_idx)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    ##############
    ## Training ##
    ##############
    print('[START TRAINING]')
    trainer = Trainer(args,
                      train_loader, val_loader, test_loader,
                      question_info_dict, answerer_info_dict, tag_to_idx,
                      device)
    trainer.train()
