import csv

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer


csvFile = open("/Users/zhouyou/Downloads/all-the-news/articles1.csv", "r")
reader = csv.reader(csvFile)



tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')

# wordnet 提取词干效果更好
p_stemmer = WordNetLemmatizer()

result = []
for item in reader:
    if reader.line_num == 1:
        continue
    else:
        # print(item[9])
        line = item[9]
        tokens = tokenizer.tokenize(line)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.lemmatize(i) for i in stopped_tokens]

    result.append(stemmed_tokens)
    break

csvFile.close()
print(result)
