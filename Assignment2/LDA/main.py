from data_preprocess import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils import load_data
from lda_gibbs import LDA_GIBBS
import argparse, os

parser = argparse.ArgumentParser(description='Latent Dirichlet Allocation')

parser.add_argument('--train_data', type=str, default='data/corpus.txt', help='train data source')
parser.add_argument('--alg', type=str, default='sklearn', help='sklearn/self')
parser.add_argument('--topic', type=int, default= 10, help='topic num')
parser.add_argument('--iter', type=int, default=100, help='training iter')
parser.add_argument('--n_top_words', type=int, default= 10, help='topic word num')


args = parser.parse_args()
if not os.path.exists(args.train_data):
    preprocess()

corpus = load_data(args.train_data)


tf_vectorizer = CountVectorizer(max_df=0.50,min_df=5,max_features=2000)
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()


if args.alg == 'sklearn':
    model = LatentDirichletAllocation(n_components=args.topic, max_iter=args.iter, learning_method='batch', n_jobs=-1)
    print("Begin training.")

    model.fit(tf)
    print(model.perplexity(tf))
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([tf_feature_names[i] for i in topic.argsort()[:-args.n_top_words-1:-1]]))
        for i in topic.argsort()[:-args.n_top_words-1:-1]:
            print("%.4f" % (topic[i] / sum(topic)), end=' ')
        print()

else:
    word2id = {}
    for idx, word in enumerate(tf_feature_names):
        word2id[word] = idx

    docs = {}
    for doc_idx, doc in enumerate(corpus):
        doc_list = [word2id[w] for w in doc.split() if w in word2id.keys()]
        docs[doc_idx] = doc_list

    print("Begin training.")
    model = LDA_GIBBS(docs, args.topic, args.iter, tf_feature_names)
    model.learn()
    print(model.get_perplexity())
    for t_id, t in enumerate(model.phi):
        print("Topic %d:" % (t_id))
        print(" ".join([tf_feature_names[i] for i in t.argsort()[:-args.n_top_words-1:-1]]))
        for i in t.argsort()[:-args.n_top_words-1:-1]:
            print("%.4f" % t[i], end=' ')
        print()