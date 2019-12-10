# 1. 看看HLDA和LLDA
# HLDA不用写K，用propagation试试
# 2. 文本数量不够的影响，这10篇内容相似
# 3. 每次结果不同，需要重复计算取平均?


# TODO: corpus加入

import os
import re
import sys

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from sklearn.cluster import AffinityPropagation
import csv

csv.field_size_limit(sys.maxsize)

document_path = "/Users/zhouyou/Downloads/doc/docsent"
summ_path = "/Users/zhouyou/Downloads/Archive/"
corpus_add = "/Users/zhouyou/Downloads/doc/corpus_add"


def readfile(path):
    # 遍历文件夹
    files = os.listdir(path)
    files.sort()
    for file in files:
        raw_text = ''
        raw_text_list = []
        if not os.path.isdir(file):
            # print("File name: " + file)
            f = open(path + "/" + file)
            for line in f.readlines():
                searchObj = re.findall(r'SNO=(.*?)>(.*?)</S>', line)
                if searchObj:
                    # print(searchObj)
                    # print(searchObj[0][0])
                    if searchObj[0][0] == "\"1\"":
                        raw_text = raw_text + searchObj[0][1] + '.' + ' '
                        raw_text_list.append(searchObj[0][1])
                    else:
                        raw_text = raw_text + searchObj[0][1] + ' '
                        raw_text_list.append(searchObj[0][1])
                    # print(s)
            yield file, raw_text, raw_text_list


# 读原文的文件.
per_raw_text = readfile(document_path)

all_raw_text = []
all_filenames = []
all_raw_text_list = []

for filename, raw_text, raw_text_list in per_raw_text:
    all_filenames.append(filename)
    all_raw_text.append(raw_text)
    all_raw_text_list.append(raw_text_list)

# print(all_raw_text)
# print(len(all_raw_text))

tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')

# p_stemmer = PorterStemmer()
# p_stemmer = SnowballStemmer('english')

# wordnet 提取词干效果更好
p_stemmer = WordNetLemmatizer()

# texts 才是最后要放入dic的list

# 目标文本加入corpus
texts = []
for i in all_raw_text:
    tokens = tokenizer.tokenize(i)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    stemmed_tokens = [p_stemmer.lemmatize(i) for i in stopped_tokens]
    # print(stemmed_tokens)
    texts.append(stemmed_tokens)
    break

# # 随机文本加入corpus
# corpus_add_files = os.listdir(corpus_add)
# firstFile = False
#
# for file in corpus_add_files:
#     try:
#         corpus_add_file = open(corpus_add + "/" + file)
#         # print(corpus_add_file)
#         for line in corpus_add_file.readlines():
#             # print(line)
#             tokens = tokenizer.tokenize(line)
#             stopped_tokens = [i for i in tokens if not i in en_stop]
#             stemmed_tokens = [p_stemmer.lemmatize(i) for i in stopped_tokens]
#
#         texts.append(stemmed_tokens)
#         # print(len(stemmed_tokens))
#         # break
#     except:
#         pass

# exit(0)

import csv
csvFile = open("/Users/zhouyou/Downloads/all-the-news/articles1.csv", "r")
reader = csv.reader(csvFile)
csv.field_size_limit(sys.maxsize)

for item in reader:
    if reader.line_num == 1:
        continue
    else:
        if reader.line_num > 2000:
            break
        # print(item[9])
        line = item[9]
        tokens = tokenizer.tokenize(line)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.lemmatize(i) for i in stopped_tokens]

    texts.append(stemmed_tokens)



print(len(texts))
# print(texts[0])
# print(texts[1])
# print("\n")
# print(len(texts))

# AP算法计算聚类中心数
ap = AffinityPropagation(preference=-50).fit()
cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_

n_clusters_ = len(cluster_centers_indices)

print(n_clusters_)

sys.exit(0)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=30, id2word=dictionary, passes=20)

# print(ldamodel.print_topics(num_topics=10, num_words=5))

doc_select = texts[0]
doc_select_list = all_raw_text_list[0]
print(all_raw_text[0])
# print(doc_select)
# print(len(all_raw_text_list))

doc_bow = dictionary.doc2bow(doc_select)

print("\n\n")

print("summary topic: ")
print(ldamodel.get_document_topics(doc_bow))
a, b = ldamodel.get_document_topics(doc_bow)[0]
# print(ldamodel.print_topic(topicno=a))
print("\n")

# for i in range(len(texts)):
#     temp = dictionary.doc2bow(texts[i])
#     print(ldamodel.get_document_topics(temp))
#
#
#
# sys.exit()


filename = all_filenames[0]
searchObj = re.findall(r'-(.*)_', filename)
extract_folder_name = searchObj[0][:-2]
file = filename[2:-7]

archive_files = os.listdir(summ_path)
archive_files.sort()

for archive_file in archive_files:
    path = summ_path + archive_file + "/extract/" + extract_folder_name
    try:
        # print("File name: " + file)
        # print(path + "/" + file + "extract")
        f = open(path + "/" + file + "extract")
        summ = ''
        for line in f.readlines():
            # print(line)
            searchObj = re.findall(r'SNO="(.*)"', line)
            if searchObj:
                # print(int(searchObj[0]))
                summ = summ + (doc_select_list[int(searchObj[0])])
        # print(archive_file + ": " + ",".join(summ) + "\n")
        print(archive_file + ": ")
        # print(summ)

        summ_tokens = tokenizer.tokenize(summ)
        summ_stopped_tokens = [i for i in summ_tokens if not i in en_stop]
        # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        summ_stemmed_tokens = [p_stemmer.lemmatize(i) for i in summ_stopped_tokens]

        # summ_corpus = [dictionary.doc2bow(text) for text in summ]

        # generate LDA model
        # summ_ldamodel = models.ldamodel.LdaModel(summ_corpus, num_topics=2, id2word=dictionary, passes=20)

        # print(summ_ldamodel.print_topics(num_topics=1, num_words=5))
        # print(summ)

        # print(summ_stemmed_tokens)

        summ_bow = dictionary.doc2bow(summ_stemmed_tokens)
        # print(summ_bow)

        print(ldamodel.get_document_topics(bow=summ_bow, minimum_probability=0.01))

        print("\n")

    except Exception as e:
        pass
