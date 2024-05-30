     #install pyvi, nltk, gensim
# import sys 
# sys.path.insert(0, '../detect_correct')

import nltk
from pyvi import ViPosTagger
import re as resub
import os
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
# from nltk.lm import KneserNeyInterpolated
from nltk.lm import MLE, NgramCounter
import pickle
import regex as re
from underthesea import pos_tag
from pyvi import ViTokenizer
from text_normalize import normalize_diacritic
from tqdm import tqdm
from nltk.util import ngrams
import time
# from vncorenlp import VnCoreNLP

# annotate=VnCoreNLP('../vncorenlp/VnCoreNLP-1.1.1.jar',annotators="wseg,pos", max_heap_size='-Xmx2g')


uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic


dicchar = loaddicchar()


def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)


def vn_word_to_telex_type(word):
    dau_cau = 0
    new_word = ''
    for char in word:
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            new_word += char
            continue
        if y != 0:
            dau_cau = y
        new_word += bang_nguyen_am[x][-1]
    new_word += bang_ky_tu_dau[dau_cau]
    return new_word


def vn_sentence_to_telex_type(sentence):
    words = sentence.split()
    for index, word in enumerate(words):
        words[index] = vn_word_to_telex_type(word)
    return ' '.join(words)



# def chuan_hoa_dau_tu_tieng_viet(word):
#     if not is_valid_vietnam_word(word):
#         return word

#     chars = list(word)
#     dau_cau = 0
#     nguyen_am_index = []
#     qu_or_gi = False
#     for index, char in enumerate(chars):
#         x, y = nguyen_am_to_ids.get(char, (-1, -1))
#         if x == -1:
#             continue
#         elif x == 9:  # check qu
#             if index != 0 and chars[index - 1] == 'q':
#                 chars[index] = 'u'
#                 qu_or_gi = True
#         elif x == 5:  # check gi
#             if index != 0 and chars[index - 1] == 'g':
#                 chars[index] = 'i'
#                 qu_or_gi = True
#         if y != 0:
#             dau_cau = y
#             chars[index] = bang_nguyen_am[x][0]
#         if not qu_or_gi or index != 1:
#             nguyen_am_index.append(index)
#     if len(nguyen_am_index) < 2:
#         if qu_or_gi:
#             if len(chars) == 2:
#                 x, y = nguyen_am_to_ids.get(chars[1])
#                 chars[1] = bang_nguyen_am[x][dau_cau]
#             else:
#                 x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
#                 if x != -1:
#                     chars[2] = bang_nguyen_am[x][dau_cau]
#                 else:
#                     chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
#             return ''.join(chars)
#         return word

#     for index in nguyen_am_index:
#         x, y = nguyen_am_to_ids[chars[index]]
#         if x == 4 or x == 8: 
#             chars[index] = bang_nguyen_am[x][dau_cau]
#             return ''.join(chars)

#     if len(nguyen_am_index) == 2:
#         if nguyen_am_index[-1] == len(chars) - 1:
#             x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
#             chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
#         else:
#             x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
#             chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
#     else:
#         x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
#         chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
#     return ''.join(chars)

def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word
    
    word = normalize_diacritic(word)
    return word

def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True

def chuan_hoa_dau_cau_tieng_viet(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        words[index] = chuan_hoa_dau_tu_tieng_viet(word)
    
    return ' '.join(words)
# def chuan_hoa_dau_cau_tieng_viet(sentence):

#     sentence = sentence.lower()
#     words = sentence.split()
#     for index, word in enumerate(words):
#         cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
#         # print(cw)
#         if len(cw) == 3:
#             cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
#         words[index] = ''.join(cw)
#     return ' '.join(words)


def load_data(line):
    train_data=[]
    # i = 0
    # for line in data:
        # i+=1
        # if i == 100:
        #     break
        # print(line)
    line=line.replace(',','.')
    line = line.replace('\n',' ')
    line = line.strip()
    line=line.split('.')
    if '' in line:
        line.remove('')
    for sentence in line:
        # sentence = sentence.lower()
        sentence=sentence.replace('\xa0',' ')
        sentence=sentence.replace('\n','')
        sentence=ViTokenizer.tokenize(sentence)
        sentence=ViPosTagger.postagging(sentence)
        new_sentence=[]
        for index,element in enumerate(sentence[1]):
            if element == 'Np':
                new_sentence.append('proper_noun')
            elif element == 'F':
                new_sentence.append('punctuation')
            elif element == 'M':
                new_sentence.append('numeral')
            elif element == 'Ny':
                new_sentence.append('noun_abbreviation')
            elif element == 'Nu':
                new_sentence.append('unit_noun')
            else:
                for ele in sentence[0][index].split('_'):
                    new_sentence.append(ele)
        new_sentence=[element.lower().rstrip('\n') for element in new_sentence]
        new_sentence=' '.join(new_sentence)
        new_sentence=chuan_hoa_dau_cau_tieng_viet(new_sentence)
        new_sentence=new_sentence.split(' ')
        if len(new_sentence) > 1:
            train_data.append(new_sentence)
    return train_data

def read_data(path, start_idx = None, length_read = None):
    lines = []
    end_idx = start_idx + length_read
    with open(path) as f:
        for line_idx, line in enumerate(f):
            if line_idx < start_idx:
                continue
            if line_idx >= end_idx:
                break
            lines.extend(load_data(line))
    
    return lines

corpus_dir = './speech_clean_text.txt'
model_name = 'n-gram.pickle'
n=1


if __name__=="__main__":
    # print('Loading data ...')
    # with open(corpus_dir,'r',encoding='utf-8') as f:
    #     train_data=load_data(f)
    # print(train_data[:10])
    # print('Training ngram model ....')
    # train_data,vocab=padded_everygram_pipeline(int(n),train_data)
    # model=MLE(int(n))
    # model.fit(train_data,vocab)
    # print('Saving ...')
    # pickle.dump(model,open(model_name,'wb'))
    
    start_idx, leng_data = 0, 100000
    model=NgramCounter()
    while True:
        print('Reading data from {}'.format(start_idx))
        train_data_list=read_data(corpus_dir, start_idx, leng_data)
        print(len(train_data_list))
        print(train_data_list[:10])
        if len(train_data_list) <= 0:
            break
        print('Training ngram model ....')
        for train_data in tqdm(train_data_list):
            # for o in range(1, int(n) + 1):
            train_data_af = pad_both_ends(train_data, n=n)
            train_data_af = ngrams(train_data_af, n)
#             print(list(train_data_af))
            model.update([train_data_af])
        start_idx = start_idx + leng_data
    with open(model_name,'wb') as wf:
        pickle.dump(model,wf)
    wf.close()
    time.sleep(2)
    
    print('Loading ngram model ....')
    with open(model_name, 'rb') as f:
        t_model = pickle.load(f)
    # print(t_model.counts)
    # print('vocab len',len(t_model.vocab))
    print('ngram count',t_model[['nguyên', 'lý', 'của']]['chiến'])
    print('count nguyen', t_model['nguyên'])
    print('count nguyen', model['nguyên'])
    
    # lines = []
    # with open('../speech_clean_text.txt') as f:
    #     for line_idx, line in enumerate(f):
    #         if line_idx >= 150:
    #             break
    #         lines.append(line)
    # print(lines)
    # with open(r'./test_clean.txt', 'w') as fp:
    #     fp.write(''.join(lines))
            
    # train_data = ['tôi không thích', 'tôi không thích cô ấy']
    # for t in train_data:
    #     train = [t.split()]
    #     print(train)
    #     train_data,vocab=padded_everygram_pipeline(int(n),train)
    #     print(vocab)
    #     model.fit(train_data,vocab)
    #     print(model.vocab['tôi'])
    #     print(model.vocab['cô'])
    #     print(model.counts[['không']]['thích'])
    #     print(model.counts[['thích']]['<UNK>'])
    #     print(model.counts)
    
    
    
    # print(chuan_hoa_dau_cau_tieng_viet('外资企业是越'))
    # print(chuan_hoa_dau_cau_tieng_viet('thủy toàn'))
    # sentence = "4 nguyên lý trong thời đại 'hóa' công nghệ số hóa . 4 nguyên lý trong thời đại công nghệ số hóa"
    # sentence=load_data([sentence])
    # print(sentence)
