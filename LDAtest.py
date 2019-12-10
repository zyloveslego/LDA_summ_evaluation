import os
import re

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models

# import nltk
# nltk.download('wordnet')


document_path = "/Users/zhouyou/Downloads/doc/docsent"
summ_path = "/Users/zhouyou/Downloads/Archive/"


def readfile(path):
    # 遍历文件夹
    files = os.listdir(path)
    files.sort()
    for file in files:
        raw_text = []
        if not os.path.isdir(file):
            # print("File name: " + file)
            f = open(path + "/" + file)
            for line in f.readlines():
                searchObj = re.findall(r'SNO=(.*?)>(.*?)</S>', line)
                if searchObj:
                    # print(searchObj)
                    # print(searchObj[0][0])
                    if searchObj[0][0] == "\"1\"":
                        raw_text.append(searchObj[0][1] + '.')
                    else:
                        raw_text.append(searchObj[0][1])
                    # print(s)

            yield file, raw_text




all_raw_text = readfile(document_path)

tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')

# p_stemmer = PorterStemmer()
# p_stemmer = SnowballStemmer('english')

# wordnet 提取词干效果更好
p_stemmer = WordNetLemmatizer()

# texts = []


for filename, raw_text in all_raw_text:
    print(filename)
    # print(len(raw_text))
    # print(raw_text)
    texts = []
    for i in raw_text:
        tokens = tokenizer.tokenize(i)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        stemmed_tokens = [p_stemmer.lemmatize(i) for i in stopped_tokens]
        # print(stemmed_tokens)
        texts.append(stemmed_tokens)

    print(texts)
    print("\n")
    # print(len(texts))

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)

    print(ldamodel.print_topics(num_topics=1, num_words=5))
    print("\n")

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
            summ = []
            for line in f.readlines():
                # print(line)
                searchObj = re.findall(r'SNO="(.*)"', line)
                if searchObj:
                    summ.append(texts[int(searchObj[0])])
            # print(archive_file + ": " + ",".join(summ) + "\n")
            print(archive_file + ": ")
            # print(summ)

            summ_corpus = [dictionary.doc2bow(text) for text in summ]

            # generate LDA model
            summ_ldamodel = models.ldamodel.LdaModel(summ_corpus, num_topics=2, id2word=dictionary, passes=20)

            print(summ_ldamodel.print_topics(num_topics=1, num_words=5))
            print("\n")

        except:
            pass


    break

# y=[2/6,3/5,3/6,3/6,5/6,5/7,4/5,1]
# x=[5,10,20,30,40,50,60,70]
