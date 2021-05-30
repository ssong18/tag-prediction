import argparse
import ast


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError
    return v


def add_math_overflow_arguments(parser):
    parser.add_argument('--answerer_removal_criterion', choices=['min', 'max'], default='max')
    parser.add_argument('--question_removal_criterion', choices=['min', 'max'], default='max')
    parser.add_argument('--max_answerers', type=int, default=5, help='number of maximum answerers per question')
    parser.add_argument('--max_questions', type=int, default=20, help='number of maximum questions per answerer')
    parser.add_argument('--max_tags', type=int, default=5, help='number of maximum tags per question for input')
    return parser


def add_stack_overflow_arguments(parser):
    parser.add_argument('--answerer_removal_criterion', choices=['min', 'max'], default='max')
    parser.add_argument('--question_removal_criterion', choices=['min', 'max'], default='max')
    parser.add_argument('--max_answerers', type=int, default=2, help='number of maximum answerers per question')
    parser.add_argument('--max_questions', type=int, default=10, help='number of maximum questions per answerer')
    parser.add_argument('--max_tags', type=int, default=5, help='number of maximum tags per question for input')
    return parser
