import argparse
import random
import os
import math
# import copy
import string
import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import nltk
import time
import evaluate
import calibration as cal
# import time
from torch import nn
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM, LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model
from huggingface_hub import snapshot_download
# from sentence_transformers import SentenceTransformer
# from datasets import load_dataset
# from sklearn.metrics import f1_score
from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel
# from collections import defaultdict
from get_task import get_task
from utils import calculate_sentence_transformer_embedding,codex_execution,expand_to_aliases
from two_steps import selective_annotation, prompt_retrieval, prompt_retrieval_random, prompt_retrieval_self_check
from get_semantics import get_similar_cluster
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', required=True,type=str)
parser.add_argument('--selective_annotation_method', required=True,type=str)
parser.add_argument('--model_cache_dir', required=True,type=str)
parser.add_argument('--data_cache_dir', required=True,type=str)
parser.add_argument('--output_dir', required=True,type=str)
parser.add_argument('--model_key', type=str)
parser.add_argument('--prompt_retrieval_method', default='similar',type=str)
parser.add_argument('--model_name', default='EleutherAI/gpt-j-6B',type=str)
parser.add_argument('--embedding_model', default='sentence-transformers/paraphrase-mpnet-base-v2',type=str)
parser.add_argument('--annotation_size', default=100,type=int)
parser.add_argument('--seed', default=0,type=int)
parser.add_argument('--batch_size', default=10,type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--maximum_input_len', default=1024)
parser.add_argument('--maximum_output_len', default=64)
parser.add_argument('--use_api_model', action="store_true")
parser.add_argument('--self_check', action="store_true")
parser.add_argument('--semantic_entropy', action="store_true")
parser.add_argument('--temperature', default=0.5, type=float)
parser.add_argument('--max_in_context_samples', default=10, type=int)
parser.add_argument('--predictive_entropy', action="store_true")
parser.add_argument('--embedding_based', action="store_true")
parser.add_argument('--only_do_generation', action="store_true")
parser.add_argument('--save_files', action='store_true')
args = parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

def roc(corrects, scores):
    auroc = metrics.roc_auc_score(corrects, scores)
    return auroc


rouge = evaluate.load('rouge')
def rouge_L(preds, golds):
    scores = []
    for pred in preds:
        ps = 0.0
        for gold in golds:
            # pred = normalize_text(pred)
            # gold = normalize_text(gold)
            rouge_scores = rouge.compute(predictions=[pred], references=[gold])['rougeL']
            if rouge_scores > ps:
                ps = rouge_scores
        scores.append(ps)
    return max(scores)

def f1_score(preds, golds):
    scores = []

    for pred in preds:
        ps = 0.0
        for gold in golds:
            ret = nltk.f_measure(set(normalize_text(pred).split()), set(normalize_text(gold).split()))
            if ret is None:
                ret = 0.0
            if ret > ps:
                ps = ret
        scores.append(ps)
    return max(scores)

def exact_match(preds, golds):
    correctness = [0]
    for pred in preds:
        for gold in golds:
            if normalize_text(pred) == normalize_text(gold):
                correctness.append(1)
                break

    return max(correctness)




def calculate_semantic_entropy(preds, probs, semantic_groups):
    ps = {}
    ts = []
    for idx, group in enumerate(semantic_groups):
        if group not in ps:
            ps[group] = torch.exp(torch.sum(probs[idx, :])) + 1e-6
        else:
            ps[group] += torch.exp(torch.sum(probs[idx, :])) + 1e-6
        ts.append(preds[idx])
    ps = [p for p in list(ps.values()) if not p == 0]
    ps = np.array(ps)
    return - np.mean(np.log(ps))

def calculate_loglikelihood(sample, predictions):
    context = sample[1]
    input_ids = tokenizer_gpt(context, return_tensors="pt").input_ids
    input_ids = input_ids[:1, :args.maximum_input_len]

    output_ids = tokenizer_gpt(predictions, padding='max_length', max_length=args.maximum_output_len, return_tensors="pt"
                                , truncation=True, add_special_tokens=False).input_ids

    # inputs = tokenizer_gpt.batch_decode(input_ids, skip_special_tokens=True)[0]
    # generations = [inputs + pred for pred in predictions]
    # output_ids = tokenizer_gpt(generations, padding='max_length', max_length=args.maximum_input_len + args.maximum_output_len, return_tensors='pt', truncation=False, return_length=True)
    output_ids = torch.cat([input_ids.repeat(output_ids.shape[0], 1), output_ids], dim=-1)

    input_len = len(input_ids[0])

    labels = output_ids.clone()
    labels[:, :input_len] = -100
    attention_mask = labels != tokenizer_gpt.pad_token_id
    labels = (1 - attention_mask.int()) * -100 + attention_mask.int() * labels
    with torch.no_grad():
        model_output = inference_model(output_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
    # Shift so that tokens < n predict n
    shift_logits = model_output.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(shift_logits.permute(0,2,1), shift_labels)
    # why this is different?
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    ln_pred_entropy = torch.mean(loss.sum(-1) / (loss != 0.0).sum(-1)).item()
    pred_entropy = torch.mean(loss.sum(-1)).item()
    return -loss, ln_pred_entropy, pred_entropy



def calculate_gen_logprob(outputs, generated_texts, tokenizer):
    logits = torch.stack(outputs.scores, dim=1)
    probs = nn.functional.softmax(logits, dim=-1)
    # for j in range(5):
    #     print(f'the jth token {j}')
    #     first_token_prob = probs[0, j, :].cpu().numpy().tolist()
    #     for i, p in enumerate(first_token_prob):
    #         if not p == 0.0:
    #             print(i)
    #             print(tokenizer.convert_ids_to_tokens([i]))
    #
    # print()
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    max_length = probs.size(1)
    encoded_output = tokenizer(generated_texts, padding='max_length', max_length=max_length, return_tensors='pt', truncation=True, return_length=True, add_special_tokens=False)

    labels = encoded_output.input_ids[:, 1:].cuda()
    pad_masks = (1 - encoded_output.attention_mask[:, 1:]).bool().cuda()

    # all_probs = probs.gather(dim=-1, index=labels).squeeze(-1)
    # all_logprobs = logprobs.gather(dim=-1, index=labels).squeeze(-1)
    all_probs = torch.gather(probs, 2, labels[:, :, None]).squeeze(-1)
    all_logprobs = torch.gather(logprobs, 2, labels[:, :, None]).squeeze(-1)
    all_probs.masked_fill_(pad_masks, 1.0)
    all_logprobs.masked_fill_(pad_masks, 0.0)
    return all_probs.prod(-1).cpu().detach().numpy().tolist(), all_probs.cpu().cpu().detach().numpy().tolist(), all_logprobs.cpu().detach().numpy().tolist()

NO_SPLIT_MODULE_CLASSES = {"facebook/opt-6.7B": "OPTDecoderLayer", "facebook/opt-13B": "OPTDecoderLayer", "EleutherAI/gpt-j-6B": "GPTJBlock", "/local2/fanyin/Llama-2-7b-hf": "LlamaDecoderLayer", "/local2/fanyin/IT_llama2/NIV2Full200": "LlamaDecoderLayer", "/local2/fanyin/Llama-2-13b-hf": "LlamaDecoderLayer", "/local2/fanyin/llama-7B": "LlamaDecoderLayer",  "openlm-research/open_llama_7b_v2": "LlamaDecoderLayer", "openlm-research/open_llama_3b_v2": "LlamaDecoderLayer"}



def self_check_lm(sample, model, tokenizer, ic_examples):
    # print('starting print')
    # print(sample)
    in_context_examples = ic_examples[0]
    context = ic_examples[1]
    one_test_example = ic_examples[2]
    # print(context)
    # print()
    # print(one_test_example)
    # print()
    # print(ic_examples[2])
    # assert(0)
    # def format_example(example, override_output=None, label='True'):
    #     answer = override_output if override_output is not None else example['label']
    #     return f"Context: {example['summary']}\nQuestion: {example['question']}\nPossible answer: {answer}\nIs the possible answer correct?\n", label
    if args.task_name == 'triviaqa' or args.task_name == 'hotpotqa':
        def format_example(example, override_output=None, label='True'):
            answer = override_output if override_output is not None else example['label']
            return f"Question: {example['question']}\nPossible answer: {answer}\nIs this answer correct?\n", label
    else:
        def format_example(example, override_output=None, label='True'):
            answer = override_output if override_output is not None else example['label']
            return f"Context: {example['summary']}\nQuestion: {example['question']}\nPossible answer: {answer}\nIs the possible answer correct?\n", label
    for predicted_answer in [ic_examples[2]['label']]:
        context = prompt_retrieval_self_check(one_test_example, context, in_context_examples, processed_train_examples, tokenizer, predicted_answer, format_example, args.maximum_input_len)
        input_ids = tokenizer_gpt(context, return_tensors="pt").input_ids
        input_ids = input_ids[:, :args.maximum_input_len]
        input_ids = input_ids.to(0)
        gen_tokens = model.generate(
            input_ids,
            do_sample=False,
            max_length=input_ids.shape[1] + 64,
            use_cache=True,
            top_k=0,
            # num_beams=3,
            # no_repeat_ngram_size=2,
            num_return_sequences=1,
            output_scores=True, return_dict_in_generate=True
        )
        generation = tokenizer_gpt.batch_decode(gen_tokens.sequences[:, len(input_ids[0]):], skip_special_tokens=True)
        generation = clean_generated_text(generation)
    return generation[0]

def load_model(args):
    if args.use_api_model:
        maximum_input_len = args.maximum_input_len
        return_string = True
        single_input_len = None
        inference_model = None
        data_module = None
        tokenizer_gpt = None
    else:
        if 'llama' in args.model_name:
            tokenizer_gpt = LlamaTokenizer.from_pretrained(args.model_name,cache_dir=args.model_cache_dir)
        else:
            tokenizer_gpt = AutoTokenizer.from_pretrained(args.model_name,cache_dir=args.model_cache_dir)
        config = AutoConfig.from_pretrained(args.model_name)
        with init_empty_weights():
            inference_model = AutoModelForCausalLM.from_config(config)

        tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
        inference_model.tie_weights()


        max_memory = {0: '16GiB', 1: '16GiB', 2: '24GiB'}
        no_split_module_classes = [NO_SPLIT_MODULE_CLASSES[args.model_name]]
        device_map = infer_auto_device_map(inference_model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
        print(device_map)
        if 'llama' in args.model_name:
            inference_model = LlamaForCausalLM.from_pretrained(args.model_name, device_map=device_map,
                                                                   cache_dir=args.model_cache_dir)
        else:
            inference_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=device_map, cache_dir=args.model_cache_dir)
        inference_model.config.pad_token_id = inference_model.config.eos_token_id
        inference_model.eval()

        data_module = None
        return_string = True
        single_input_len = None
        maximum_input_len = args.maximum_input_len
    in_context_kwargs = {'inference_data_module': data_module, 'return_string': return_string,
                         'single_input_len': single_input_len, 'maximum_input_len': maximum_input_len}
    return inference_model, tokenizer_gpt, in_context_kwargs


def calculate_entropy(lens, samples, log_samples):
    num_samples = 0
    hs = []
    print(samples) 
    for idx, sample in enumerate(samples):
        prob = np.prod(sample)
        sample.reverse()
        s_l = lens[idx]
        if prob == 0:
            continue
        if s_l == 0:
            continue
        num_samples += 1
        h = - math.log(prob)
        hs.append(h)
    if not num_samples == 0:
        return sum(hs) / num_samples
    else:
        return -100

def clean_generated_text(generated_texts):
    stop = ['--', '</s>', '<unk>', '\n', ';', '#', "'Question'"]
    prediction = []
    for idx, generated_text in enumerate(generated_texts):
        stop_index = len(generated_text)
        for i, c in enumerate(generated_text):
            if c.strip(' ') in stop:
                stop_index = i
                break
        prediction.append(generated_text[:stop_index])
    return prediction

if __name__=='__main__':
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)


    # load inference model
    inference_model, tokenizer_gpt, kwargs = load_model(args)


    # load data and compute in-context examples

    train_examples, eval_examples, train_text_to_encode, eval_text_to_encode, format_example, label_map = get_task(args=args)

    if args.embedding_based:

        total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=train_text_to_encode,
                                                                      args=args)
        total_eval_embeds = calculate_sentence_transformer_embedding(text_to_encode=eval_text_to_encode,
                                                                      args=args)



        if os.path.isfile(os.path.join(args.output_dir,'first_phase_selected_indices.json')):
            with open(os.path.join(args.output_dir,'first_phase_selected_indices.json')) as f:
                first_phase_selected_indices = json.load(f)
        else:
            first_phase_selected_indices = selective_annotation(embeddings=total_train_embeds,
                                                                train_examples=train_examples,
                                                                format_example=format_example,
                                                                label_map=label_map,
                                                                inference_model=inference_model,
                                                                tokenizer_gpt=tokenizer_gpt,
                                                                args=args,
                                                                **kwargs
                                                                )
            with open(os.path.join(args.output_dir, 'first_phase_selected_indices.json'),'w') as f:
                json.dump(first_phase_selected_indices, f, indent=4)

        processed_train_examples = [train_examples[idx] for idx in first_phase_selected_indices]
        processed_eval_examples = eval_examples

        prompt_retrieval(tokenizer=tokenizer_gpt,
                         train_embs=total_train_embeds[first_phase_selected_indices],
                         test_embs=total_eval_embeds,
                         train_examples=processed_train_examples,
                         eval_examples=eval_examples,
                         format_example=format_example,
                         label_map=label_map,
                         args=args,
                         **kwargs)

        prompt_cache_dir = os.path.join(args.output_dir, 'prompts')
        candidate_prompt_files = os.listdir(prompt_cache_dir)
        prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
        assert len(prompt_files) == len(processed_eval_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                                  f"len(processed_eval_examples)={len(processed_eval_examples)}"
    else:
        processed_train_examples = train_examples
        processed_eval_examples = eval_examples

        prompt_retrieval_random(tokenizer=tokenizer_gpt,
                         train_examples=processed_train_examples,
                         eval_examples=eval_examples,
                         format_example=format_example,
                         label_map=label_map,
                         args=args,
                         **kwargs)

        prompt_cache_dir = os.path.join(args.output_dir, 'prompts')
        candidate_prompt_files = os.listdir(prompt_cache_dir)
        prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
        assert len(prompt_files) == len(processed_eval_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                                  f"len(processed_eval_examples)={len(processed_eval_examples)}"

    output_dir = os.path.join(args.output_dir, 'results')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # start generation
    golds = []
    preds = []

    contexts = []
    incontext_examples = []
    sampled_generations = []
    verifications = []

    execution_count = 0

    for file in tqdm(prompt_files, total=len(prompt_files), desc=f"  LLM inference"):
        if not os.path.isfile(os.path.join(output_dir,file)):
            if not args.use_api_model:
                with open(os.path.join(prompt_cache_dir, file)) as f:
                    one_test_example = json.load(f)
                context = one_test_example[1]
                in_context_example = '\n'.join(context.split('\n')[:-1])
                incontext_examples.append(in_context_example)
                print(context)
                input_ids = tokenizer_gpt(context, return_tensors="pt").input_ids
                input_len = len(input_ids[0])
                input_ids = input_ids[:, :args.maximum_input_len]
                # print(input_ids.size(1))
                input_ids = input_ids.to(0)
                with torch.no_grad():
                    generated_texts = []

                    generations = []
                    score = []
                    prob_list = []
                    logprob_list = []
                    for i in range(1):
                        gen_tokens = inference_model.generate(
                            input_ids,
                            do_sample=False,
                            max_length=input_ids.shape[1] + 64,
                            use_cache=True,
                            top_p=args.top_p,
                            top_k=0,
                            temperature=args.temperature,
                            num_return_sequences=1,
                            output_scores=True, return_dict_in_generate=True
                        )
                        generation = tokenizer_gpt.batch_decode(gen_tokens.sequences[:, len(input_ids[0]):], skip_special_tokens=False)
                        generation = clean_generated_text(generation)
                        print(generation)
                        print(one_test_example[2]['label'])
                        if not args.only_do_generation:
                            generations += tokenizer_gpt.batch_decode(gen_tokens.sequences[:, :], skip_special_tokens=True)
                            s, prob, logprob = calculate_gen_logprob(gen_tokens, generation,
                                                                           tokenizer_gpt)
                            score += s
                            prob_list += prob
                            logprob_list += logprob
                        generated_texts += generation
                s = self_check_lm(generation, inference_model, tokenizer_gpt, one_test_example)
                verifications.append(s)
                contexts.append(context)
                golds.append(one_test_example[2]['label'])
                sampled_generations.append(generated_texts)
                if not args.only_do_generation:
                    with open(f"{output_dir}/{file}", 'w') as f:
                        json.dump(
                            [generated_texts, generations,
                             score, prob_list, logprob_list, input_len], f, indent=4)
                # else:
                #     print(contexts[-1])
                #     print(golds[-1])
                #     print(sampled_generations[-1])


            else:
                cur_key = model_keys[execution_count % len(model_keys)]
                execution_count += 1
                try:
                    codex_execution(key=cur_key, output_path=os.path.join(output_dir, file),
                                    prompt_path=os.path.join(prompt_cache_dir, file))
                except Exception as e:
                    print(e)
                    time.sleep(3)


    # start evaluation
    if args.task_name=='xsum':
        preds = []
        golds = []
        for file in prompt_files:
            with open(os.path.join(prompt_cache_dir, file)) as f:
                one_test_example = json.load(f)
                gold = one_test_example[2]['summary']
            with open(os.path.join(output_dir, file)) as f:
                pred = json.load(f)
            preds.append(pred[1])
            golds.append(gold)

        assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
        preds, golds = postprocess_text(preds, golds)
        metric = load_metric("rouge")
        result = metric.compute(predictions=preds, references=golds, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        with open(os.path.join(args.output_dir,'result_summary.json'), 'w') as f:
            json.dump(result, f)
        print(result)
    elif args.task_name=='nq':
        correct = 0
        total = 0
        for file in prompt_files:
            with open(os.path.join(prompt_cache_dir, file)) as f:
                one_test_example = json.load(f)
            answers = expand_to_aliases(one_test_example[2]["long"] + one_test_example[2]["short_targets"],
                                        make_sub_answers=True)
            with open(os.path.join(output_dir, file)) as f:
                pred_dict = json.load(f)
            prediction = pred_dict['choices'][0]['text'].replace('\n', ' ')
            prediction = ' '.join(prediction.split(' ')[1:])
            predictions = expand_to_aliases([prediction])
            if len(list(answers & predictions)) > 0:
                correct += 1
            total += 1
        with open(os.path.join(args.output_dir,'result_summary.txt'), 'w') as f:
            f.write(f"{total} examples, accuracy is: {correct / total}\n")
        print(f"{total} examples, accuracy is: {correct / total}\n")
    elif args.task_name == 'narraqa' or args.task_name == 'triviaqa' or args.task_name == 'coqa' or args.task_name == 'hotpotqa':
        f1_sum = 0
        total = 0
        logprobs = []
        probs = []
        f1s = []
        ems = []
        shs = []
        hs = []
        ln_hs = []
        se_hs = []
        ss = []

        sampled_generations = []
        gold_generations = []
        contexts = []
        ids = []
        for idx, file in tqdm(enumerate(prompt_files), total=len(prompt_files)):
            if idx > 2000:
                break
            with open(os.path.join(prompt_cache_dir, file)) as f:
                one_test_example = json.load(f)
            with open(os.path.join(output_dir, file)) as f:
                pred = json.load(f)
            if args.task_name == 'narraqa':
                candidate1 = one_test_example[2]['answer1']
                candidate2 = one_test_example[2]['answer2']
                golds = [candidate1, candidate2]
            else:
                golds = [one_test_example[2]['label']]

            predictions = [p for p in pred[0] if not len(p.strip()) == 0]
            if len(predictions) == 0:
                print(idx)
                print(pred[0])
                continue
            ids.append(idx)
            # lens = [len(l) for l in tokenizer_gpt(predictions, add_special_tokens=False)['input_ids']]
            if args.self_check:
                s = self_check_lm(pred, inference_model, tokenizer_gpt, one_test_example)
                ss.append(s)
            if args.predictive_entropy:
                # est_h = calculate_entropy(lens, pred[3], pred[4])
                loss, ln_est_h, est_h = calculate_loglikelihood(one_test_example, predictions)
                hs.append(-est_h)
                ln_hs.append(-ln_est_h)
            if args.semantic_entropy:
                semantic_cluster = get_similar_cluster(one_test_example[2]['question'], predictions)
                semantic_est_h = calculate_semantic_entropy(predictions, loss, semantic_cluster)
                se_hs.append(-semantic_est_h)


            # em = exact_match(predictions, golds)
            # f1 = f1_score(predictions, golds)
            # rouge_l = rouge_L(predictions, golds)
            rouge_l = rouge.compute(predictions=[predictions[0]], references=[golds[0]])['rougeL']
            em = 1 if rouge_l >= 0.5 else 0

            # f1s.append(f1)

            total += 1
            ems.append(em)
            contexts.append(one_test_example[1])
            sampled_generations.append(predictions)
            gold_generations.append(golds)

        if args.save_files:
            df = pd.DataFrame({"context": contexts, 'gold_answer': gold_generations, 'sampled_answers': sampled_generations})
            df.to_csv(os.path.join(output_dir, 'prepared_data.csv'), encoding='ascii', errors='replace', index=False)
            df = pd.DataFrame({"labels": ems})
            df.to_csv(os.path.join(output_dir, 'labels.csv'), encoding='ascii', errors='replace', index=False)
        print(ids)
        print(len(ids))
        print(hs)
        print(ln_hs)
        print(ems)
        accuracy = sum(ems) / total
        print(accuracy)
        auroc = roc(ems, hs)
        print('predictive entropy', auroc)
        ln_auroc = roc(ems, ln_hs)
        print('length-normalized predictive entropy', ln_auroc)
        se_auroc = roc(ems, se_hs)
        print('semantic predictive entropy', se_auroc)
        accuracy = sum(ems) / total
        with open(os.path.join(args.output_dir,'result_summary.txt'), 'w') as f:
            f.write(f"ln_auroc: {ln_auroc}, auroc: {auroc}, se_auroc: {se_auroc}, {total} examples, em is: {accuracy}\n")
