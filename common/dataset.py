import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from collections import defaultdict


def remove_answerers_from_question(question_info_dict, answerer_info_dict, max_answerers, removal_criterion):
    """
    If number of answerers exceed the max answerers for each question,
        remove the answerers who have the (small or large) number of replies.
    """
    new_question_info_dict = defaultdict(list)
    for question_idx, answerers in question_info_dict.items():
        if len(answerers) > max_answerers:
            questions_per_answerers = []
            for answerer_idx in answerers:
                answerer_info = answerer_info_dict[answerer_idx]
                questions_per_answerers.append(len(answerer_info))
            if removal_criterion == 'max':
                max_questions_idx = np.array(questions_per_answerers).argsort()[-max_answerers:]
                new_answerers = [answerers[i] for i in max_questions_idx]
            elif removal_criterion == 'min':
                min_questions_idx = np.array(questions_per_answerers).argsort()[:max_answerers]
                new_answerers = [answerers[i] for i in min_questions_idx]
            else:
                raise NotImplementedError
            new_question_info_dict[question_idx] = new_answerers
        else:
            new_question_info_dict[question_idx] = answerers

    return new_question_info_dict


def remove_questions_from_answerer(answerer_info_dict, question_info_dict, max_questions, removal_criterion):
    """
    If number of questions exceed the max questions for each answerer,
        remove the questions which have the (small or large) number of answerers.
    """
    new_answerer_info_dict = defaultdict(list)
    for answerer_idx, questions in answerer_info_dict.items():
        if len(questions) > max_questions:
            answerers_per_questions = []
            for question_idx in questions:
                question_info = question_info_dict[question_idx]
                answerers_per_questions.append(len(question_info))
            if removal_criterion == 'max':
                max_answerers_idx = np.array(answerers_per_questions).argsort()[-max_questions:]
                new_questions = [questions[i] for i in max_answerers_idx]
            elif removal_criterion == 'min':
                min_answerers_idx = np.array(answerers_per_questions).argsort()[:max_questions]
                new_questions = [questions[i] for i in min_answerers_idx]
            else:
                raise NotImplementedError
            new_answerer_info_dict[answerer_idx] = new_questions
        else:
            new_answerer_info_dict[answerer_idx] = questions

    return new_answerer_info_dict


class TagPredictionTrainDataset(Dataset):
    def __init__(self, args, data, question_info_dict, answerer_info_dict, question_to_tags, tag_to_idx):
        self.args = args
        self.data = data
        self.question_info_dict = question_info_dict
        self.answerer_info_dict = answerer_info_dict
        self.question_to_tags = question_to_tags
        self.tag_to_idx = tag_to_idx

    def __len__(self):
        return len(self.data['question_idxes'])

    def _build_input(self, index):
        A = self.args.max_answerers
        Q = self.args.max_questions
        T = self.args.max_tags

        aPAD = len(self.answerer_info_dict)
        question_idx = self.data['question_idxes'][index]
        question_by_answerer_ids = self.question_info_dict[question_idx]
        question_by_answerer_ids += [aPAD] * (A - len(question_by_answerer_ids))

        qPAD = len(self.question_info_dict)
        question_by_question_ids = []
        for answerer_idx in question_by_answerer_ids:
            if answerer_idx != aPAD:
                question_ids = self.answerer_info_dict[answerer_idx]
                question_ids += [qPAD] * (Q - len(question_ids))
            else:
                question_ids = [qPAD] * Q
            question_by_question_ids.append(question_ids)

        tPAD = len(self.tag_to_idx)
        question_by_tag_ids = []
        for question_ids in question_by_question_ids:
            for question_id in question_ids:
                if (question_id != qPAD) and question_id in self.question_to_tags:
                    tags = self.question_to_tags[question_id]
                    tag_ids = [self.tag_to_idx[tag] for tag in tags]
                    tag_ids += [tPAD] * (T - len(tag_ids))
                else:
                    tag_ids = [tPAD] * T
                question_by_tag_ids.append(tag_ids)

        question_by_tag_ids = torch.LongTensor(question_by_tag_ids).reshape(A, Q, T)

        # General cross-entropy or kl-divergence is not feasible since there exists more than 6000 distinct tags.
        # Therefore, select one positive tag and N negative tags and apply mutual information loss
        tag_ids = self.data['labels'][index]
        tag_ids = [self.tag_to_idx[tag] for tag in tag_ids]
        pos_tag_id = np.random.choice(tag_ids)
        neg_tag_ids = np.random.choice(np.setdiff1d(np.arange(tPAD), tag_ids),
                                       self.args.negative_samples,
                                       replace=False)
        tag_ids = np.concatenate(([pos_tag_id], neg_tag_ids))
        tag_ids = torch.LongTensor(tag_ids)

        targets = torch.zeros_like(tag_ids)
        targets[0] = 1

        return (question_by_tag_ids, tag_ids), targets

    def __getitem__(self, index):
        return self._build_input(index)


class TagPredictionEvalDataset(TagPredictionTrainDataset):
    def __init__(self, args, data, question_info_dict, answerer_info_dict, question_to_tags, tag_to_idx):
        super(TagPredictionEvalDataset, self).__init__(
            args, data, question_info_dict, answerer_info_dict, question_to_tags, tag_to_idx)

    def __getitem__(self, index):
        (question_by_tag_ids, _), _ = self._build_input(index)

        T = self.args.max_tags
        tPAD = len(self.tag_to_idx)
        tag_ids = self.data['labels'][index]
        tag_ids = [self.tag_to_idx[tag] for tag in tag_ids]

        targets = F.one_hot(torch.LongTensor(tag_ids), num_classes=len(self.tag_to_idx))
        targets = targets.sum(dim=0)
        tag_ids = torch.arange(len(self.tag_to_idx))
        tag_ids = torch.LongTensor(tag_ids)

        return (question_by_tag_ids, tag_ids), targets
