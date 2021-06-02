from collections import defaultdict
import pickle


if __name__ == "__main__":
    data_types = ['mathoverflow', 'stackoverflow']
    for data_type in data_types:
        data = {}
        answerer_info_dict = defaultdict(list)  # {answerer: [q1, q3, ..., qn]}
        question_info_dict = defaultdict(list)  # {question: [a2, a5, ..., an]}
        question_to_tags = defaultdict(list)  # {question: [t1, t45, .., tn]}
        tag_to_idx = {}
        train_data = {'question_idxes': [],
                      'labels': defaultdict(list)}
        val_data = {'question_idxes': [],
                    'labels': defaultdict(list)}
        test_data = {'question_idxes': [],
                     'labels': defaultdict(list)}

        # 1. Build answerer_info_dict & question_info_dict
        with open("data/"+data_type+'/answerer_info_'+data_type+'.txt') as f:
            answerer_info = f.read()
        answerer_info = answerer_info.split('\n')[:-1]
        for answerer_idx, questions in enumerate(answerer_info):
            questions = list(map(int, questions.split(' ')))
            answerer_info_dict[answerer_idx] = questions
            for question in questions:
                question_info_dict[question].append(answerer_idx)

        # 2. Build train & val & test dataset
        # 2.1. Inputs
        with open("data/"+data_type+'/train_query_'+data_type+'.txt') as f:
            train_query = f.read()
        train_query = list(map(int, train_query.split('\n')[:-1]))
        train_data['question_idxes'] = train_query

        with open("data/"+data_type+'/valid_query_'+data_type+'.txt') as f:
            val_query = f.read()
        val_query = list(map(int, val_query.split('\n')[:-1]))
        val_data['question_idxes'] = val_query

        with open("data/"+data_type+'/test_query_'+data_type+'.txt') as f:
            test_query = f.read()
        test_query = list(map(int, test_query.split('\n')[:-1]))
        test_data['question_idxes'] = test_query

        # 2.2. Labels
        with open("data/"+data_type+'/train_answer_'+data_type+'.txt') as f:
            train_answer = f.read()
        train_answer = train_answer.split('\n')[:-1]
        for answer_idx, tags in enumerate(train_answer):
            tags = list(map(int, tags.split(' ')))
            question = train_query[answer_idx]
            question_to_tags[question] = tags
            train_data['labels'][answer_idx] = tags
            for tag in tags:
                if tag not in tag_to_idx:
                    tag_to_idx[tag] = len(tag_to_idx)

        with open("data/"+data_type+'/valid_answer_'+data_type+'.txt') as f:
            val_answer = f.read()
        val_answer = val_answer.split('\n')[:-1]
        for answer_idx, tags in enumerate(val_answer):
            tags = list(map(int, tags.split(' ')))
            question = train_query[answer_idx]
            question_to_tags[question] = tags
            val_data['labels'][answer_idx] = tags
            for tag in tags:
                if tag not in tag_to_idx:
                    tag_to_idx[tag] = len(tag_to_idx)

        # fill the empty values
        for answer_idx in range(len(test_query)):
            test_data['labels'][answer_idx] = []

        data['question_info_dict'] = question_info_dict
        data['answerer_info_dict'] = answerer_info_dict
        data['question_to_tags'] = question_to_tags
        data['tag_to_idx'] = tag_to_idx
        data['train'] = train_data
        data['val'] = val_data
        data['test'] = test_data

        with open(data_type + '.pickle', 'wb') as f:
            pickle.dump(data, f)
