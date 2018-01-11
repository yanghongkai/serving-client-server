# !/usr/bin/env python
# coding=utf-8
import pypinyin
import pickle as pkl
import os
import time
import numpy as np
import operator

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations

from parsing.config import TF_SERVING_HOST, TF_SERVING_PORT

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(this_dir, "..")
# print("data_dir:{}".format(data_dir))


class BaseTokenizer(object):

    def __init__(self):  # FIXME
        pass

    def cut(self, sents):  # FIXME
        raise NotImplementedError("Abstract Method")


class CharacterTokenizer(BaseTokenizer):

    def __init__(self):  # FIXME
        self.num = 1

    def cut(self, sents):
        self.num += 1
        line=[]
        tags=[]
        tag_dic={'0':'tag0','1':'tag1','2':'tag2'}
        word_tags=sents.split(' ')
        for word_tag in word_tags:
            tmp=word_tag.split('/')
            word=tmp[0]
            tag=[]
            #print("word:{}".format(word))
            pinyin=pypinyin.lazy_pinyin(word,0)[0]
            line.append(pinyin)
            for i in pinyin:
                tag.append(tag_dic[tmp[1]])
            tag=' '.join(tag)
            tags.append(tag)
            tags.append('tag0')

        tags=tags[:-1]
        response_tag=' '.join(tags)
        sents=' '.join(line)
        response = ' '.join([i.replace(' ', '_space') for i in sents])
        return response+' _link '+response_tag

    def cut_f(self, input_file, output_file):
        fin = open(input_file, 'r')
        fout = open(output_file, 'w')
        for l in fin:
            fout.write(self.cut(l.strip()) + '\n')
        fin.close()
        fout.close()


class Classify():
    def __init__(self, model_name, signature_name, encode=None):
        self.model_name = model_name
        self.signature_name = signature_name
        self.tokenizer = CharacterTokenizer()
        self.vocab_dict = self.load_vocab_dict()
        self.label_dict = self.load_label_dict()
        self.encode = encode
        self.gate = {0: "", 1: "东门", 2: "南门", 3: "西门", 4: "北门", 5: "后门"}
        self.part = {0: "", 1: "凌晨", 2: "早上", 3: "中午", 4: "下午", 5: "晚上"}
        self.type = {0: "录像", 1: "监控", 2: "预案", 3: "视频"}

    def classify(self, sents):
        sents = sents.strip()
        self.sents = self.sents2id(sents)
        self.inp_w, self.inp_t, self.inp_sl = self.read_data(self.sents)

        # hostport = '192.168.31.186:6000'
        # grpc
        # host, port = hostport.split(':')
        # channel = implementations.insecure_channel(host, int(port))
        channel = implementations.insecure_channel(TF_SERVING_HOST, TF_SERVING_PORT)

        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        # build request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        request.inputs['input_plh'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.inp_w, dtype=tf.int32))
        request.inputs['input_t'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.inp_t, dtype=tf.int32))
        request.inputs['input_sl'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.inp_sl, dtype=tf.int32))

        request.inputs['dropout_keep_prob_mlp'].CopyFrom(
            tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))
        model_result = stub.Predict(request, 60.0)
        # model_result = stub.Predict.future(request, 60.0)
        # print(self.signature_name, model_result)
        model_result = np.array(model_result.outputs['scores'].float_val)
        #print("model_result:{}".format(model_result))
        index, _ = max(enumerate(model_result), key=operator.itemgetter(1))
        if index > 0:
            label = self.label_dict[index-1]
        else:
            label = ""
        if self.encode == "door" and label:
            label = self.gate[label]
        if self.encode == "part":
            if label:
                label = self.part[label]
            else:
                label = self.part[0]
        if self.encode == "type":
            if label:
                label = self.type[label]
            else:
                label = self.part[0]
        return label

    def load_vocab_dict(self):
        vocab_path = os.path.join(data_dir, "vocab", self.model_name+"_vocab_inword.pkl")
        with open(vocab_path, "rb") as f:
            vocab_dict = pkl.load(f)
        return vocab_dict

    def load_label_dict(self):
        label_path = os.path.join(data_dir, "label", self.model_name+"_label_class_mapping.pkl")
        with open(label_path, "rb") as f:
            label_dict = pkl.load(f)
        label_dict = {v:k for k,v in label_dict.items()}
        return label_dict

    def sents2id(self, sents):
        # print("sents:{}".format(sents))
        sents = self.tokenizer.cut(sents)
        # print("tokenizer sents:{}".format(sents))
        sents2id = [self.vocab_dict.get(word, self.vocab_dict['OOV']) for word in sents.split(' ')]
        return sents2id

    def read_data(self, l):
        dictionary = {v: k for k, v in self.vocab_dict.items()}
        tmp_w = []
        tmp_t = []
        tmp_l = l
        tagsplit = 0
        for k, v in dictionary.items():
            if v == '_link':
                tagsplit = k
                # print("taglist k:{}".format(tagsplit))
                # taglist k: 4
                break

        tmp_w = [int(w) for w in tmp_l[:len(tmp_l) // 2]]
        tmp_t = [int(t) - tagsplit - 1 for t in tmp_l[len(tmp_l) // 2 + 1:]]

        for i in range(4):
            tmp_w.insert(0, 0)
            tmp_w.append(0)

            tmp_t.insert(0, 0)
            tmp_t.append(0)

        tmp_len = len(tmp_w)

        return [tmp_w], [tmp_t], [tmp_len]


def parse_cnn_tag(sents, model_name, encode=None):
   cls = Classify(model_name, model_name, encode)
   p_label = cls.classify(sents)
   return p_label


if __name__ == "__main__":
    st = time.perf_counter()
    sents = "请/0 调/0 出/0 汇/0 坤/0 园/0 北/0 门/0 二/1 零/1 一/1 六/1 年/1 四/0 月/0 九/0 日/0 中/0 午/0 十/0 一/0 点/0 四/0 十/0 分/0 的/0 录/0 像/0 四/0 倍/0 速/0 回/0 放/0"
    p_label = parse_cnn_tag(sents, "year")
    print("sents:{}\tp_year:{}".format(sents, p_label))
    print(time.perf_counter() - st)

    sents = "查/0 看/0 去/0 年/0 五/1 月/1 九/0 日/0 凌/0 晨/0 零/0 点/0 十/0 四/0 分/0 到/0 一/0 点/0 三/0 十/0 三/0 分/0 新/0 华/0 携/0 程/0 商/0 贸/0 市/0 场/0 东/0 门/0 的/0 录/0 像/0 负/0 三/0 倍/0 速/0"
    p_label = parse_cnn_tag(sents, "month")
    print("sents:{}\tp_month:{}".format(sents, p_label))
    print(time.perf_counter() - st)

    sents = "请/0 调/0 出/0 去/0 年/0 十/0 一/0 月/0 二/1 十/1 四/1 日/1 凌/0 晨/0 一/0 点/0 十/0 三/0 分/0 到/0 二/0 点/0 二/0 十/0 一/0 分/0 新/0 疆/0 广/0 播/0 电/0 视/0 大/0 学/0 北/0 门/0 的/0 录/0 像/0 一/0 倍/0 速/0"
    p_label = parse_cnn_tag(sents, "day")
    print("sents:{}\tp_day:{}".format(sents, p_label))
    print(time.perf_counter() - st)

    sents = "我/0 要/0 看/0 乌/0 鲁/0 木/0 齐/0 市/0 第/0 四/0 十/0 九/0 中/0 学/0 东/0 门/0 去/0 乌/0 鲁/0 木/0 齐/0 推/0 拿/0 职/0 业/0 学/0 校/0 南/0 门/0 沿/0 西/0 虹/0 东/0 路/0 的/0 监/0 控/0 "
    p_label = parse_cnn_tag(sents, "part", encode="part")
    print("sents:{}\tp_part:{}".format(sents, p_label))
    print(time.perf_counter() - st)

    sents = "查/0 看/0 去/0 年/0 五/0 月/0 九/0 日/0 凌/0 晨/0 零/0 点/0 十/0 四/0 分/0 到/0 一/0 点/0 三/0 十/0 三/0 分/0 新/0 华/0 携/0 程/0 商/0 贸/0 市/0 场/0 东/0 门/0 的/0 录/0 像/0 负/1 三/1 倍/1 速/1"
    p_label = parse_cnn_tag(sents, "speed")
    print("sents:{}\tp_speed:{}".format(sents, p_label))
    print(time.perf_counter() - st)

    sents = "查/0 看/0 海/0 德/0 酒/0 店/0 去/0 天/0 百/0 货/0 特/0 莱/0 斯/0 南/0 门/0 沿/0 农/0 大/0 东/0 路/0 的/0 监/1 控/1"
    p_label = parse_cnn_tag(sents, "type", encode="type")
    print("sents:{}\tp_type:{}".format(sents, p_label))
    print(time.perf_counter() - st)

