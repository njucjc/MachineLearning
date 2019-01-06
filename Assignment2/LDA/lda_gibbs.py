import numpy as np

"""
参考：https://github.com/fishermanff/ParallelGibbsLda/blob/master/src/main/java/org/eric/gibbs/serial/SerialGibbsLda.java
"""
class LDA_GIBBS(object):
    def __init__(self, documents, k, max_iter, vocab, alpha=2, beta=0.5):

        # document matrix
        self.documents = documents
        # num of topics
        self.k = k
        self.alpha = alpha
        self.beta = beta
        # max iter
        self.max_iter = max_iter
        # vocab
        self.vocab = vocab
        # vocab size
        self.v = len(self.vocab)
        # num of documents
        self.m = len(self.documents)
        # nd[m][k]: number of words in document m assigned to topic k; size M * K
        self.nd = np.zeros((self.m, self.k))
        # ndsum[m]: total number of words in document m; size M
        self.ndsum = np.zeros(self.m)
        # nw[i][k]: number of instances of word i assigned to topic k; size V * K
        self.nw = np.zeros((self.v, self.k))
        # nwsum[k]: total number of words assigned to topic k; size K
        self.nwsum = np.zeros(self.k)
        # z[m][i]: the assigned topic of word i in document m
        self.z = {}
        # store estimated theta matrix
        self.theta = np.zeros((self.m, self.k))
        # store estimated phi matrix
        self.phi = np.zeros((self.k, self.v))
        # store perplexity
        self.perplexity = -1

        for doc_id, doc in self.documents.items():
            self.z[doc_id] = [0] * len(doc)
            for i in range(len(doc)):
                t = np.random.randint(0, self.k)
                self.z[doc_id][i] = t
                self.nd[doc_id, t] += 1
                self.nw[self.documents[doc_id][i], t] += 1
                self.nwsum[t] += 1
            self.ndsum[doc_id] = len(doc)

    def _gibbs_sampling(self):
        for step in range(self.max_iter):

            if step % 10 == 0:
                print("Sampling step: [" + str(step + 1) + "/" + str(self.max_iter) + "]....")
                
            for i in range(self.m):
                for j in range(len(self.documents[i])):
                    self.nd[i, self.z[i][j]] -= 1
                    self.ndsum[i] -= 1
                    self.nw[self.documents[i][j], self.z[i][j]] -= 1
                    self.nwsum[self.z[i][j]] -= 1

                    prob = np.zeros(self.k)
                    for t in range(self.k):
                        prob[t] = (self.nd[i, t] + self.alpha) / (self.ndsum[i] + self.k * self.alpha) * \
                                (self.nw[self.documents[i][j], t] + self.beta) / (self.nwsum[t] + self.v * self.beta)

                    for t in range(1, self.k):
                        prob[t] += prob[t - 1]

                    r = np.random.uniform(0, prob[self.k - 1])
                    for t in range(self.k):
                        if r < prob[t] :
                            self.z[i][j] = t
                            break
                    # update counter
                    self.nd[i, self.z[i][j]] += 1
                    self.ndsum[i] += 1
                    self.nw[self.documents[i][j], self.z[i][j]] += 1
                    self.nwsum[self.z[i][j]] += 1

        self._calc_theta()
        self._calc_phi()
        
    def _calc_theta(self):
        for i in range(self.m):
            for t in range(self.k):
                self.theta[i, t] = (self.nd[i, t] + self.alpha) / (self.ndsum[i] + self.k * self.alpha)
    
    def _calc_phi(self):
        for j in range(self.v):
            for t in range(self.k):
                self.phi[t, j] = (self.nw[j, t] + self.beta) / (self.nwsum[t] + self.v * self.beta)

    def get_perplexity(self):
        if self.perplexity != -1:
            return self.perplexity
        num = 0
        den = 0
        for i in range(self.m):
            for j in range(len(self.documents[i])):
                t = self.z[i][j]
                num -= np.log(self.theta[i, t] * self.phi[t, self.documents[i][j]])
            den += len(self.documents[i])
        
        return np.exp(num / den)

    def learn(self):
        self._gibbs_sampling()

