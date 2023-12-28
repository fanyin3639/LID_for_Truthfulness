import argparse
import csv
import os
import pickle
import random

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()

meteor = evaluate.load('meteor')

deberta_predictions = []

def get_similar_cluster(question, generated_texts):
    unique_generated_texts = list(set(generated_texts))

    answer_list_1 = []
    answer_list_2 = []
    has_semantically_different_answers = False
    inputs = []
    syntactic_similarities = {}
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    for rouge_type in rouge_types:
        syntactic_similarities[rouge_type] = 0.0

    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    # print(semantic_set_ids)
    # print('Number of unique answers:', len(unique_generated_texts))

    if len(unique_generated_texts) > 1:

        # Evalauate semantic similarity
        for i, reference_answer in enumerate(unique_generated_texts):
            for j in range(i + 1, len(unique_generated_texts)):

                answer_list_1.append(unique_generated_texts[i])
                answer_list_2.append(unique_generated_texts[j])

                qa_1 = question + ' ' + unique_generated_texts[i]
                qa_2 = question + ' ' + unique_generated_texts[j]

                input = qa_1 + ' [SEP] ' + qa_2
                inputs.append(input)
                encoded_input = tokenizer.encode(input, padding=True)
                prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                deberta_prediction = 1
                # print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                if 0 in predicted_label or 0 in reverse_predicted_label:
                    has_semantically_different_answers = True
                    deberta_prediction = 0

                else:
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

                deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])

        rouge = evaluate.load('rouge')

        # Evalauate syntactic similarity
        answer_list_1 = []
        answer_list_2 = []
        for i in generated_texts:
            for j in generated_texts:
                if i != j:
                    answer_list_1.append(i)
                    answer_list_2.append(j)
        results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
        #
        # for rouge_type in rouge_types:
        #     syntactic_similarities[rouge_type] = results[rouge_type].mid.fmeasure

    # result_dict[id_] = {
    #     'syntactic_similarities': syntactic_similarities,
    #     'has_semantically_different_answers': has_semantically_different_answers
    # }
    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    return list_of_semantic_set_ids
    # result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids
