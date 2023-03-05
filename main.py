# -*- coding: utf-8 -*-
# code follow  Shaowei Chen, chenshaowei0507@163.com

import time
import gc
import torch
from alphabet import Alphabet
from entityRelation import entityRelation
import sys
import numpy as np
import random
import os
import logging
import logging.config
import argparse
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed_num = 57
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


class model_param(object):
    def __init__(self, word_alphabet, label_alphabet, relation_alphabet, char_alphabet, pretrain_word_embedding,
                 embedding_dim):
        self.word_alphabet = word_alphabet
        self.label_alphabet = label_alphabet
        self.relation_alphabet = relation_alphabet
        self.char_alphabet = char_alphabet
        self.pretrain_word_embedding = pretrain_word_embedding
        self.embedding_dim = embedding_dim
        self.pretrain_char_embedding = None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 tokens,
                 token_ids,
                 token_mask,
                 chars,
                 char_ids,
                 char_mask,
                 charLength,
                 tokenLength,
                 labels,
                 label_ids,
                 relations,
                 gold_relations):
        self.tokens = tokens
        self.token_ids = token_ids
        self.token_mask = token_mask
        self.tokenLength = tokenLength
        self.labels = labels
        self.label_ids = label_ids
        self.relations = relations
        self.gold_relations = gold_relations
        self.chars = chars
        self.char_ids = char_ids
        self.char_mask = char_mask
        self.charLength = charLength


#### target token level precision ####
def targetPredictCheck(targetPredict, batch_target_label, mask):
    pred = targetPredict.cpu().data.numpy()
    gold = batch_target_label.cpu().data.numpy()
    mask = mask.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


#### relation token level precision ####
def relationPredictCheck(relationPredict, batch_relation, all_input_mask):
    batch_size = relationPredict.size(0)
    seq_len = relationPredict.size(3)
    relation_num = relationPredict.size(1)
    maskTemp1 = all_input_mask.view(batch_size, 1, 1, seq_len).repeat(1, relation_num, seq_len, 1)
    maskTemp2 = all_input_mask.view(batch_size, 1, seq_len, 1).repeat(1, relation_num, 1, seq_len)
    maskMatrix = (maskTemp1 * maskTemp2).cpu().data.numpy()
    relationCheck = torch.zeros_like(relationPredict) + 0.1
    pred = relationPredict.cpu()
    pred = torch.gt(pred, relationCheck.cpu()).data.numpy()
    gold = batch_relation.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * gold)
    right_no_token = np.sum(overlaped * (1 - gold) * maskMatrix)
    total_token = gold.sum()
    total_no_relation = ((1 - gold) * maskMatrix).sum()
    return right_token, total_token, right_no_token, total_no_relation


def recover_label(targetPredict, all_labels, all_input_mask):  # pred_tag(batch, max_seqlen in batch)
    pred_variable = targetPredict
    gold_variable = all_labels
    mask_variable = all_input_mask
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    pred_label = []
    gold_label = []


    for idx in range(batch_size):
        pred = [pred_tag[idx][idy] - 1 for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [gold_tag[idx][idy] - 1 for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def evalForBIO(goldResult, predResult):  # 0,1,2
    correct = 0
    predicted = 0
    relevant = 0
    # count correct
    for idx in range(len(goldResult)):
        gold = goldResult[idx]
        pred = predResult[idx]
        predicted += pred.count(1)
        relevant += gold.count(1)
        for num in range(len(gold)):
            if gold[num] == 1:
                if num < len(gold) - 1:
                    if gold[num + 1] != gold[num] + 1:
                        if pred[num] == gold[num] and pred[num + 1] != gold[num] + 1:
                            correct += 1
                    else:
                        if pred[num] == gold[num]:
                            for j in range(num + 1, len(gold)):
                                if gold[j] == gold[num] + 1:
                                    if pred[j] == gold[num] + 1:
                                        if j == len(gold) - 1:
                                            correct += 1
                                            break
                                    else:
                                        break
                                else:
                                    if pred[j] != gold[num] + 1:
                                        correct += 1
                                    break
                else:
                    if pred[num] == 1:
                        correct += 1
    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1

def cout_rel_class(pred_relations, gold):
    web_test_index = pd.read_csv('Web_test_relation_class_idx.csv') # test_index
    web_train_index = pd.read_csv('web_relation_class_idx.csv') # train_index

    

    
    web_test_index = web_test_index.values.tolist()  
    g_relations = np.zeros([int(args.relationNum),1])
    p_relations = np.zeros([int(args.relationNum),1])
    c_relations = np.zeros([int(args.relationNum),1])

    print(g_relations.shape)

    print(f'total sentences :{len(pred_relations)}')

    for idx in range(len(pred_relations)):
        standard = gold[idx].relations
        pred = pred_relations[idx]    
        for i in standard :
            # print(i[4]-1)
            g_relations[i[4]]+=1
        for i in pred :
            p_relations[i[4]]+=1
        for r in standard:
            if r in pred:
                c_relations[r[4]]+=1
    

    precision = c_relations/ (p_relations + 1e-6)
    recall = c_relations/ (g_relations+ 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    
    rel_data = pd.DataFrame({'g_relations':g_relations.tolist(),'p_relations':p_relations.tolist(),'c_relations':c_relations.tolist(),'precision':precision.tolist(),'recall':recall.tolist(),'f1':f1.tolist()})
    if int(args.relationNum) ==211 :
        rel_data.to_csv('WebNLG_relation_class.csv')
    else:
        rel_data.to_csv('NYT_relation_class.csv')


def cout_trip_num(pred_relations, gold):
    gold_1, gold_2, gold_3, gold_4, gold_5 =[], [], [], [], []
    pre_1, pre_2, pre_3, pre_4, pre_5 =[], [], [], [], []

    for idx in range(len(pred_relations)):
        standard = gold[idx].relations
        pred = pred_relations[idx]   
        num = len(standard)
        if num==1:
            gold_1.append(standard)
            pre_1.append(pred)
        if num==2:
            gold_2.append(standard)
            pre_2.append(pred)
        if num==3:
            gold_3.append(standard)
            pre_3.append(pred)
        if num==4:
            gold_4.append(standard)
            pre_4.append(pred)
        if num>=5:
            gold_5.append(standard)
            pre_5.append(pred)


    pre1, rec1, f11 = prf_cal(pre_1, gold_1)
    pre2, rec2, f12 = prf_cal(pre_2, gold_2)
    pre3, rec3, f13 = prf_cal(pre_3, gold_3)
    pre4, rec4, f14 = prf_cal(pre_4, gold_4)
    pre5, rec5, f15 = prf_cal(pre_5, gold_5)

    print(f'1 triplets precision: {pre1} | recall: {rec1} | f1: {f11}')
    print(f'2 triplets precision: {pre2} | recall: {rec2} | f1: {f12}')
    print(f'3 triplets precision: {pre3} | recall: {rec3} | f1: {f13}')
    print(f'4 triplets precision: {pre4} | recall: {rec4} | f1: {f14}')
    print(f'5 triplets precision: {pre5} | recall: {rec5} | f1: {f15}')





def cout_trip_class(pred_relations, gold):
    rel_normal = []
    rel_single = []
    rel_pair = []
    rel_sp_single=[]

    pre_normal = []
    pre_single = []
    pre_pair = []
    pre_sp_single=[]

    for idx in range(len(pred_relations)):
        standard = gold[idx].relations
        pred = pred_relations[idx]   
        s_flag = 0
        sp_flag = 0
        p_flag = 0
        triplet_s = []

        for i in standard:
            for j in triplet_s:
                if i[0:4] == j[0:4]: # EPO
                    p_flag = 1
                if i[0:2] == j[0:2] and i[2:4] !=j[2:4]  or i[2:4]==j[0:2] : # p_Norm:3260, p_EPO:969, p_SEO:1299, p_sp_SEO:492
                    s_flag = 1
                    if i[0:2] == j[0:2] and i[4] == j[4]:
                        sp_flag=1
            triplet_s.append(i)




        if p_flag == 0 and s_flag == 0:
            rel_normal.append(standard)
            pre_normal.append(pred)
        if p_flag ==1:
            rel_pair.append(standard)
            pre_pair.append(pred)
        if s_flag ==1:
            rel_single.append(standard)
            pre_single.append(pred)
        if sp_flag ==1:
            rel_sp_single.append(standard)
            pre_sp_single.append(pred)


    print(f'number of p_Norm:{len(pre_normal)}, p_EPO:{len(pre_pair)}, p_SEO:{len(pre_single)}, p_sp_SEO:{len(pre_sp_single)}')
    print(f'number of g_Norm:{len(rel_normal)}, g_EPO:{len(rel_pair)}, g_SEO:{len(rel_single)}, g_sp_SEO:{len(rel_sp_single)}')

    Norm_pre, Norm_rec, Norm_f1 = prf_cal(pre_normal, rel_normal)
    EPO_pre, EPO_rec, EPO_f1 = prf_cal(pre_pair, rel_pair)
    SEO_pre, SEO_rec, SEO_f1 = prf_cal(pre_single, rel_single)
    sp_SEO_pre , sp_SEO_rec, sp_SEO_f1 = prf_cal(pre_sp_single, rel_sp_single)

    print(f'Norm triplets precision: {Norm_pre} | recall: {Norm_rec} | f1: {Norm_f1}')
    print(f'EPO triplets precision: {EPO_pre} | recall: {EPO_rec} | f1: {EPO_f1}')
    print(f'SEO triplets precision: {SEO_pre} | recall: {SEO_rec} | f1: {SEO_f1}')
    print(f'sp_SEO triplets precision: {sp_SEO_pre} | recall: {sp_SEO_rec} | f1: {sp_SEO_f1}')





def fmeasure_overlap(pred_relations, gold):

    goldTotal = 0
    OverlapTotal = 0
    predictTotal = 0
    for idx in range(len(pred_relations)):
        standard = gold[idx].relations
        pred = pred_relations[idx]
        goldTotal += len(standard)
        predictTotal += len(pred)
        for r1 in standard:
            for r2 in pred:
                if ifRelationOverlap(r1, r2):
                    OverlapTotal += 1
                    break
    precision = float(OverlapTotal) / (predictTotal + 1e-6)
    recall = float(OverlapTotal) / (goldTotal + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


def fmeasure_strict(pred_relations, gold):
    goldTotal = 0
    correct = 0
    predictTotal = 0
    for idx in range(len(pred_relations)):
        standard = gold[idx].relations
        pred = pred_relations[idx]
        goldTotal += len(standard)
        predictTotal += len(pred)
        for r in standard:
            if r in pred:
                correct += 1
    precision = float(correct) / (predictTotal + 1e-6)
    recall = float(correct) / (goldTotal + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print(f"{precision}, | {recall}, |{f1}")
    return precision, recall, f1


def ifRelationOverlap(relation1, relation2):
    em1_overlap = True
    em2_overlap = True
    type_correct = (relation1[-1] == relation2[-1])
    if relation1[0] >= relation2[1] or relation1[1] <= relation2[0]:
        em1_overlap = False
    if relation1[2] >= relation2[3] or relation1[3] <= relation2[2]:
        em2_overlap = False
    return em1_overlap and em2_overlap and type_correct


def prf_cal(prediction,gold):

    goldTotal = 0
    correct = 0
    predictTotal = 0
    for idx in range(len(prediction)):
        standard = gold[idx]
        pred = prediction[idx]
        goldTotal += len(standard)
        predictTotal += len(pred)
        for r in standard:
            if r in pred:
                correct += 1
    precision = float(correct) / (predictTotal + 1e-6)
    recall = float(correct) / (goldTotal + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1



def make_relation(R_tensor, instance_text, thred):  # R_tensor (relation_num, seqlen,seqlen)

    total_result = []
    for idx in range(len(instance_text)): # 对于每个句子
        entityList = []
        relationResult = []
        tempR = R_tensor[idx]
        for idy in range(len(instance_text[idx])): # 对于句子中的每个单词
            if instance_text[idx][idy] == 1:  # 如果元素值为1
                if idy == len(instance_text[idx]) - 1:  # 如果单词为句子的尾单词
                    entityList.append([idy, idy + 1]) # 加入实体
                else: # 如果不是尾单词
                    for k in range(idy + 1, len(instance_text[idx])):  # 对元素为1的单词之后的序列进行遍历
                        if instance_text[idx][k] != instance_text[idx][idy] + 1:  # 如果不等于2
                            entityList.append([idy, k])  # 得到实体范围，加入实体
                            break  # 跳出来，继续对idy进行遍历
                        elif instance_text[idx][k] == instance_text[idx][idy] + 1 and k == len(instance_text[idx]) - 1:  # 如果结尾的元素标记是2
                            entityList.append([idy, k + 1])  # 得到实体范围，加入实体
                            break
        
        for e1 in entityList:
            for e2 in entityList:
                for idy in range(len(tempR)):
                    # score = np.sum(tempR[idy, e1[0]:e1[1], e2[0]:e2[1]]) / ((e1[1] - e1[0]+1)*(e2[1] - e2[0]+1))
                    # score = np.max(tempR[idy, e1[0]:e1[1], e2[0]:e2[1]])
                    score = np.mean(tempR[idy, e1[0]:e1[1], e2[0]:e2[1]])


                    if score > thred:
                        if [e1[0], e1[1], e2[0], e2[1], idy] not in relationResult:
                            relationResult.append([e1[0], e1[1], e2[0], e2[1], idy + 1])
        total_result.append(relationResult)
    return total_result


def evaluate(test_set,model,args, output_file_path, mode):
    pred_results = []
    gold_results = []
    relation_result = []

    # set model in eval model
    model.eval()

    batch_size = args.batchSize
    test_num = len(test_set)
    total_batch = test_num // batch_size + 1
    for step in range(total_batch):
        start = step * batch_size
        end = (step + 1) * batch_size
        if end > test_num:
            end = test_num

        # make batch input data
        batch = test_set[start:end]
        if len(batch) == 0:
            continue
        all_input_ids, input_length, input_recover, all_input_mask, all_char_ids, char_length, char_recover, char_mask, all_relations, all_labels = make_data(
            batch, args)
        targetPredict, relationPredict = model(all_input_ids, input_length, all_input_mask, all_char_ids,
                                               char_length, char_recover)

        # with open('codo_info.txt','a') as f:
        #     f.write('relationPredict size:{}\n'.format(relationPredict.size()))
        # sys.exit()

        # get real label
        # pred_label, gold_label = recover_label(targetPredict[input_recover], all_labels[input_recover],
        #                                        all_input_mask[input_recover])
        # with open ('codo_info.txt','a') as f:
        #     f.write('input length: {}'.format(input_length[input_recover]))
        # sys.exit()
        pred_label, gold_label = recover_label(targetPredict[input_recover], all_labels[input_recover],
                                               all_input_mask[input_recover])

        pred_results += pred_label
        gold_results += gold_label
        relation_result += list(relationPredict[input_recover].cpu().data.numpy())

    assert args.tagScheme == "BIO"
    TP, TR, TF = evalForBIO(gold_results, pred_results)
    pred_relations = make_relation(relation_result, pred_results, args.relationThred)
    P_OverlapW, R_OverlapW, F_OverlapW = fmeasure_overlap(pred_relations, test_set)
    PW, RW, FW = fmeasure_strict(pred_relations, test_set)

    labelDic = ["O", "B", "I", "O"]
    output_file = open(output_file_path, "w", encoding="utf-8")
    for k in range(len(pred_results)):
        words = test_set[k].tokens
        pred = pred_results[k]
        gold = gold_results[k]
        relations = pred_relations[k]
        if len(words) != len(pred):
            print(len(words))
            print(len(pred))
        if len(words) != len(gold):
            print(len(words))
            print(len(gold))
        for j in range(len(gold)):
            output_file.write(words[j] + "\t" + labelDic[gold[j]] + "\t" + labelDic[pred[j]] + "\n")
        output_file.write("#Relations\n")
        for r in relations:
            output_file.write(
                str(r[0]) + "\t" + str(r[1]) + "\t" + str(r[2]) + "\t" + str(r[3]) + "\t" + str(r[4]) + "\n")
        output_file.write("\n")
    output_file.close()
    return P_OverlapW, R_OverlapW, F_OverlapW, PW, RW, FW, TP, TR, TF


def make_data(train_features, args):
    all_input_ids = torch.tensor([f.token_ids for f in train_features], dtype=torch.long)
    batchSize = all_input_ids.size(0)
    all_input_mask = torch.tensor([f.token_mask for f in train_features], dtype=torch.long)
    input_length = torch.tensor([f.tokenLength for f in train_features], dtype=torch.long).view(batchSize)
    all_labels = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
    all_char_ids = torch.tensor([f.char_ids for f in train_features], dtype=torch.long)
    char_length = torch.tensor([f.charLength for f in train_features], dtype=torch.long)
    char_mask = torch.tensor([f.char_mask for f in train_features], dtype=torch.long)
    seqLen = torch.max(input_length)
    charLen = torch.max(char_length)
    all_input_ids = all_input_ids[:, :seqLen]
    all_input_mask = all_input_mask[:, :seqLen]
    all_labels = all_labels[:, :seqLen]
    char_length = char_length[:, :seqLen]
    char_length = char_length + char_length.eq(0).long()
    all_char_ids = all_char_ids[:, :seqLen, :charLen]
    char_mask = char_mask[:, :seqLen, :charLen]
    all_relations = torch.zeros(batchSize, args.relationNum, seqLen, seqLen).long()
    for idx in range(len(train_features)):
        relations = train_features[idx].relations
        for r in relations:
            all_relations[idx, r[-1] - 1, r[0]:r[1], r[2]:r[3]] = torch.ones(r[1] - r[0], r[3] - r[2]).long()

    input_length, word_perm_idx = input_length.sort(0, descending=True)
    all_input_ids = all_input_ids[word_perm_idx]
    all_input_mask = all_input_mask[word_perm_idx]
    all_labels = all_labels[word_perm_idx]
    all_char_ids = all_char_ids[word_perm_idx]
    char_length = char_length[word_perm_idx]
    char_mask = char_mask[word_perm_idx]
    all_relations = all_relations[word_perm_idx]

    all_char_ids = all_char_ids.view(int(batchSize) * int(seqLen), -1)
    char_mask = char_mask.view(int(batchSize) * int(seqLen), -1)
    char_length = char_length.view(int(batchSize) * int(seqLen), )
    char_length, char_perm_idx = char_length.sort(0, descending=True)
    all_char_ids = all_char_ids[char_perm_idx]
    char_mask = char_mask[char_perm_idx]

    _, char_recover = char_perm_idx.sort(0, descending=False)
    _, input_recover = word_perm_idx.sort(0, descending=False)

    if args.ifgpu:
        all_input_ids = all_input_ids.cuda()
        all_input_mask = all_input_mask.cuda()
        all_labels = all_labels.cuda()
        all_relations = all_relations.cuda()
        input_length = input_length.cuda()
        input_recover = input_recover.cuda()
        all_char_ids = all_char_ids.cuda()
        char_length = char_length.cuda()
        char_recover = char_recover.cuda()
        char_mask = char_mask.cuda()

    return all_input_ids, input_length, input_recover, all_input_mask, all_char_ids, char_length, char_recover, char_mask, all_relations, all_labels


def main(args):
    # make dir
    if not os.path.exists(args.test_eval_dir):
        os.makedirs(args.test_eval_dir)
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    logging.basicConfig(filename=args.log_dir +'_' + str(args.loss_weight) + ".log", level=logging.INFO)

    # print configs
    print(args)

    # read data
    logging.info("#### Loading dataset ####")
    datasets = torch.load(args.data)
    train_set = datasets["train"]
    test_set = datasets["test"]
    valid_set = datasets["dev"]
    word_alphabet = datasets["word_alpha"]
    label_alphabet = datasets["label_alpha"]
    relation_alphabet = datasets["relation_alpha"]
    char_alphabet = datasets["char_alpha"]

    # load pretrain embedding
    logging.info("#### Loading pretrain embeddings ####")
    pretrain = torch.load(args.pretrain_emb)
    pretrain_word_embedding = pretrain["preTrainEmbedding"]
    embedding_dim = pretrain["emb_dim"]

    model_params = model_param(word_alphabet, label_alphabet, relation_alphabet, char_alphabet, pretrain_word_embedding, embedding_dim)

    # defined model
    logging.info("#### Building model ####")
    model = entityRelation(args, model_params)
    # if test
    if args.mode == "test":
        assert args.test_model != ""
        model = torch.load(args.test_model)
        P_OverlapW, R_OverlapW, F_OverlapW, PW, RW, FW, TP, TR, TF = evaluate(test_set, model, args,
                                                                              args.test_eval_dir + "/test_output",
                                                                              "test")
        test_start = time.time()
        test_finish = time.time()
        test_cost = test_finish - test_start
        logging.info("test: time: %.2fs" % test_cost)
        logging.info("relation strict: Precision: %.4f; Recall: %.4f; F1: %.4f" % (PW, RW, FW))
        logging.info("relation overlap: Precision: %.4f; Recall: %.4f; F1: %.4f" % (P_OverlapW, R_OverlapW, F_OverlapW))
        logging.info("entity: Precision: %.4f; Recall: %.4f; F1: %.4f" % (TP, TR, TF))
    else:
        # Building optimizer
        logging.info("#### Building optimizer ####")
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in param_optimizer if "encoder" in n or "relation" in n or "Embedding" in n]}]
        optimizer = torch.optim.RMSprop(optimizer_grouped_parameters,
                                        lr=args.lr_rate)
        optimizer1 = torch.optim.RMSprop(optimizer_grouped_parameters1,
                                         lr=args.R_lr_rate)

        best_Score_val = -10000
        best_Score_test = -10000

        lr = args.lr_rate
        r_lr = args.R_lr_rate
        # start training
        for idx in range(args.iteration):
            epoch_start = time.time()
            temp_start = epoch_start
            logging.info("Epoch: %s/%s" % (idx, args.iteration))

            # adjust learning rate
            if idx > 2:
                lr = args.lr_decay * lr
                r_lr = args.lr_decay * r_lr
                optimizer.param_groups[0]["lr"] = lr
                optimizer1.param_groups[0]["lr"] = r_lr
                logging.info(lr)

            sample_loss = 0
            total_loss = 0
            right_target_token = 0
            whole_target_token = 0
            right_relation_token = 0
            whole_relation_token = 0
            right_noRelation_token = 0
            whole_noRelation_token = 0

            # set model in train model
            model.train()
            model.zero_grad()
            #
            random.shuffle(train_set)
            batch_size = args.batchSize
            train_num = len(train_set)
            total_batch = train_num // batch_size + 1
            for step in range(total_batch):
                start = step * batch_size
                end = (step + 1) * batch_size
                if end > train_num:
                    end = train_num

                # make batch input data
                batch = train_set[start:end]
                if len(batch) == 0:
                    continue
                all_input_ids, input_length, input_recover, all_input_mask, all_char_ids, char_length, char_recover, char_mask, all_relations, all_labels = make_data(
                    batch, args)

                tloss, rloss, targetPredict, relationPredict = model.neg_log_likelihood_loss(all_input_ids,
                                                                                             input_length,
                                                                                             all_input_mask,
                                                                                             all_char_ids, char_length,
                                                                                             char_recover,
                                                                                             all_relations, all_labels)

                # check right number
                targetRight, targetWhole = targetPredictCheck(targetPredict, all_labels, all_input_mask)
                relationRight, relationWhole, right_no_token, total_no_relation = relationPredictCheck(
                    relationPredict, all_relations, all_input_mask)

                # cal right and whole label number
                right_target_token += targetRight
                whole_target_token += targetWhole
                right_relation_token += relationRight
                whole_relation_token += relationWhole
                right_noRelation_token += right_no_token
                whole_noRelation_token += total_no_relation

                # cal loss
                sample_loss += rloss.item() + tloss.item()
                total_loss += rloss.item() + tloss.item()


                # print train info
                if step % 300 == 0 :
                    temp_time = time.time()
                    temp_cost = temp_time - temp_start
                    temp_start = temp_time
                    logging.info(
                        "     Instance: %s; Time: %.2fs; loss: %.4f; target_acc: %s/%s=%.4f; relation_acc: %s/%s=%.4f; noRelation_acc: %s/%s=%.10f" % (
                            step * args.batchSize, temp_cost, sample_loss, right_target_token, whole_target_token,
                            (right_target_token + 0.) / whole_target_token, right_relation_token, whole_relation_token,
                            (right_relation_token + 0.) / whole_relation_token, right_noRelation_token,
                            whole_noRelation_token, (right_noRelation_token + 0.) / whole_noRelation_token))
                    if sample_loss > 1e9 or str(sample_loss) == "nan":
                        logging.info(
                            "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                        exit(1)
                    sys.stdout.flush()  # update output show
                    sample_loss = 0

                if step % 2 == 0:
                    loss = args.loss_weight*rloss + (1-args.loss_weight)*tloss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    rloss.backward()
                    optimizer1.step()
                    optimizer1.zero_grad()

            temp_time = time.time()
            temp_cost = temp_time - temp_start
            logging.info(
                "     Instance: %s; Time: %.2fs; loss: %.4f; target_acc: %s/%s=%.4f; relation_acc: %s/%s=%.4f; noRelation_acc: %s/%s=%.10f" % (
                    step * args.batchSize, temp_cost, sample_loss, right_target_token, whole_target_token,
                    (right_target_token + 0.) / whole_target_token, right_relation_token, whole_relation_token,
                    (right_relation_token + 0.) / whole_relation_token, right_noRelation_token, whole_noRelation_token,
                    (right_noRelation_token + 0.) / whole_noRelation_token))

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            logging.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
                idx, epoch_cost, len(train_set) / epoch_cost, total_loss))
            logging.info("totalloss:", total_loss)
            if total_loss > 1e8 or str(total_loss) == "nan":
                logging.info("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                exit(1)

            if idx % 5==0:
                # test evaluate
                P_OverlapW, R_OverlapW, F_OverlapW, PW, RW, FW, TP, TR, TF = evaluate(
                    test_set, model, args, args.eval_dir + "/test_output_" + str(idx), "test")
                test_finish = time.time()
                # test_cost = test_finish - dev_finish
                current_Score_test = FW

                # logging.info("test: time: %.2fs" % test_cost)
                logging.info("relation strict: %.4f; Recall: %.4f; F1: %.4f" % (PW, RW, FW))
                logging.info(
                    "relation overlap: Precision: %.4f; Recall: %.4f; F1: %.4f" % (P_OverlapW, R_OverlapW, F_OverlapW))
                logging.info("entity result: Precision: %.4f; Recall: %.4f; F1: %.4f" % (TP, TR, TF))
                gc.collect()
                if current_Score_test > best_Score_test:
                    logging.info("Exceed previous best f score with entity f: %.4f and relation f: %.4f  in test" % (
                        TF, FW))
                    best_Score_test = current_Score_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])

    parser.add_argument('--data', type=str, default="./data/WebNLG.pt")
    # test
    parser.add_argument('--test_model', type=str, default="./model/WebNLG/modelFinal.model")
    parser.add_argument('--test_eval_dir', type=str, default="./test_eval/WebNLG")

    # train
    parser.add_argument('--pretrain_emb', type=str, default="./data/WebNLG_embedding.pt")
    parser.add_argument('--model_dir', type=str, default="./model/WebNLG/")
    parser.add_argument('--eval_dir', type=str, default="./eval/WebNLG")
    parser.add_argument('--log_dir', type=str, default="./log/WebNLG")


    parser.add_argument('--tagScheme', type=str, default="BIO")
    parser.add_argument('--relationNum', type=int, default=211, choices=[24, 211])

    parser.add_argument('--useChar', type=bool, default=True)
    parser.add_argument('--charExtractor', type=str, default="LSTM", choices=["LSTM", "CNN"])
    parser.add_argument('--char_embedding_dim', type=int, default=50)
    parser.add_argument('--char_hidden_dim', type=int, default=100)
    parser.add_argument('--encoderExtractor', type=str, default="LSTM")
    parser.add_argument('--encoder_Bidirectional', type=bool, default=True)
    parser.add_argument('--encoder_dim', type=int, default=600)
    parser.add_argument('--targetHiddenDim', type=int, default=250)
    parser.add_argument('--relationHiddenDim', type=int, default=70) # nyt 8 | webnlg:70
    parser.add_argument('--relationThred', type=float, default=0.8)  # nyt :0.6 | webnlg : 0.8
    parser.add_argument('--relationLinearpara', type=int, default=100)
    parser.add_argument('--l1_weight', type=float, default=0)  # l1_loss weight
    



    parser.add_argument('--ifgpu', type=bool, default=True)
    parser.add_argument('--iteration', type=int, default=50)
    parser.add_argument('--batchSize', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr_rate', type=float, default=0.001)
    parser.add_argument('--R_lr_rate', type=float, default=0.002)
    parser.add_argument('--lr_decay', type=float, default=0.95)

    parser.add_argument('--loss_weight', type=float, default=0.5)
    

    args = parser.parse_args()
    main(args)
