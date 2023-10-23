import time
import numpy
import scipy.special
import re


class LDAVI:
    """
    Latent dirichlet allocation,
    Blei, David M and Ng, Andrew Y and Jordan, Michael I, 2003

    Latent Dirichlet allocation with Variational EM
    """

    class Document:
        """
        Document class for this
        """

        def __init__(self, wordids, wordcts):
            self.wordids = wordids
            self.wordcts = wordcts
            self.N = len(wordids)

    def __init__(self, num_topics, doc_file_, vocas, output_dir_name, alpha=0.1, eta=0.01, small_num=1e-100):
        """
        Initialize parameters and variables
        Do not modify this function
        :param num_topics: Number of topics
        :param doc_file_path: documents formed BOW
        :param vocas: words list
        :param output_dir_name: output directory name
        :param alpha: alpha value
        :param eta: eta value
        :param small_num: small number for avoid divide by zero
        :return: void
        """
        self._K = num_topics
        self._words = vocas
        self._output_dir_name = output_dir_name
        self._docs = doc_file_
        
        self._W = len(self._words)
        self._D = len(self._docs)
        self._alpha = alpha
        self._eta = eta
        self._small_num = small_num

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = numpy.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = self._dirichlet_expectation(self._lambda)
        self._expElogbeta = numpy.exp(self._Elogbeta)

        self._gamma = numpy.random.gamma(100., 1./100., (self._D, self._K))

    @staticmethod
    def _dirichlet_expectation(alpha):
        """
        Compute E[log(theta)] given alpha
        Compute E[log(beta)] given lambda
        Do not modify this function
        :param alpha: alpha value or lambda value
        :return: E[log(theta)] or E[log(beta)]
        """
        if 1 == len(alpha.shape):
            return scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha))
        return scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha, 1))[:, numpy.newaxis]

    def _e_step(self, e_max_iter, meanchangethresh):
        """
        Run E step in variational EM
        Update phi and gamma and
        Compute sufficient statistics for updating lambda (sstats) for M step
        :param e_max_iter: maximum iterations for e step
        :param meanchangethresh: minimum value for checking convergence
        :return: sufficient statistics for updating lambda
        """
        Elogtheta = self._dirichlet_expectation(self._gamma)
        expElogtheta = numpy.exp(Elogtheta)
        sstats = numpy.zeros(self._lambda.shape)

        for doc_idx, doc in enumerate(self._docs):
            print(doc_idx)
            ids = doc.wordids
            cts = doc.wordcts
            gammad = self._gamma[doc_idx, :]
            expElogthetad = expElogtheta[doc_idx, :]
            expElogbetad = self._expElogbeta[:, ids]

            for _ in range(e_max_iter):
                lastgamma = gammad
                words_phi=expElogthetad.reshape(-1,1) * expElogbetad
                # normalization phi for each word
                sum_of_phis_for_each_words= words_phi.sum(axis=0)
                words_phi_normalization= (words_phi/sum_of_phis_for_each_words).T # 제가 가로세로 헷갈려서 뒤집어놓습니다;;
                words_phi_weighted=numpy.array(cts).reshape(-1,1)*words_phi_normalization
                sum_of_pih_50 = self._alpha + numpy.sum(words_phi_weighted,axis=0)
                gammad =  sum_of_pih_50
                
                meanchange = numpy.mean(abs(gammad - lastgamma))
                if meanchange < meanchangethresh:
                    break
            self._gamma[doc_idx, :] = gammad
        # for문이 도는동안 sstat을 게속 더해주는게 훨씬 나을 것같아 for 문 안에 작성하였습니다.
            sstats[:,ids]=sstats[:,ids]+words_phi_weighted.T

        return sstats

    def _m_step(self, sstats):
        """
        Run M step in variational EM
        Update lambda
        Do not modify this function
        :param sstats: sufficient statistics for updating lambda from E step
        :return: None
        """
        self._lambda = self._eta + sstats

        self._Elogbeta = self._dirichlet_expectation(self._lambda)
        self._expElogbeta = numpy.exp(self._Elogbeta)

        return None

    def run(self, max_iter=10, e_max_iter=100, echangethresh=1e-6, emchangethresh=1e-6, do_print_log=False):
        """
        Run variational EM algorithm
        Do not modify this function
        :param max_iter: maximum iterations for EM
        :param e_max_iter: maximum iterations for e step
        :param echangethresh: minimum value for checking convergence in e step
        :param emchangethresh: minimum value for checking convergence for EM
        :param do_print_log: Do we print the result on each iteration?
        :return: void
        """
        num_words_docs = 0
        last_estimated_perp = 0
        for doc in self._docs:
            num_words_docs += sum(doc.wordcts)
        if do_print_log:
            prev = time.time()

        for idx in range(max_iter):
            sstats = self._e_step(e_max_iter, echangethresh)
            self._m_step(sstats)

            estimated_perp = self._compute_perplexity(num_words_docs)
            meanchange = numpy.mean(abs(estimated_perp - last_estimated_perp))
            last_estimated_perp = estimated_perp

            if do_print_log:
                print('{}\t{:.2f} sec\theld-out perplexity:\t{:.2f}'.format(idx, time.time() - prev, estimated_perp))
                prev = time.time()

            if meanchange < emchangethresh:
                break

        return None

    def _compute_perplexity(self, num_words_docs):
        """
        Compute held-out perplexity
        Do not modify this function
        :param num_words_docs:
        :return:
        """
        bound = self._approx_bound() / num_words_docs
        estimated_perp = numpy.exp(-bound)

        return estimated_perp

    def _approx_bound(self):
        """
        Approximate the bound for perplexity
        Do not modify this function
        :return: bound
        """
        score = 0
        Elogtheta = self._dirichlet_expectation(self._gamma)

        # E[log p(docs | theta, beta)]
        for doc_idx, doc in enumerate(self._docs):
            ids = doc.wordids
            cts = doc.wordcts
            phinorm = numpy.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[doc_idx, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = numpy.log(sum(numpy.exp(temp - tmax))) + tmax
            score += numpy.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += numpy.sum((self._alpha - self._gamma) * Elogtheta)
        score += numpy.sum(scipy.special.gammaln(self._gamma) - scipy.special.gammaln(self._alpha))
        score += sum(scipy.special.gammaln(self._alpha * self._K) - scipy.special.gammaln(numpy.sum(self._gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score += numpy.sum((self._eta - self._lambda) * self._Elogbeta)
        score += numpy.sum(scipy.special.gammaln(self._lambda) - scipy.special.gammaln(self._eta))
        score += numpy.sum(scipy.special.gammaln(self._eta * self._W) - scipy.special.gammaln(numpy.sum(self._lambda, 1)))

        return score

    def export_result(self, output_file_name, rank_idx=10):
        """
        Export Algorithm Result to File
        Do not modify this function
        :param output_file_name: output file name
        :param rank_idx: how many topics are printed?
        :return: void
        """
        # Raw data
        numpy.savetxt("{}/{}_gamma.csv".format(self._output_dir_name, output_file_name), self._gamma, delimiter=",")
        numpy.savetxt("{}/{}_lambda.csv".format(self._output_dir_name, output_file_name), self._lambda, delimiter=",")

        # Ranked data
        with open("{}/{}_topics_Ranked.csv".format(self._output_dir_name, output_file_name), "w") as ranked_topic_word_file:
            row_idx = -1

            for each_row in self._lambda:
                row_idx += 1
                ranked_one = sorted(enumerate(each_row), key=lambda x: x[1], reverse=True)
                print('topic {},{}'.format(row_idx,
                                           ",".join([self._words[x[0]] for x in ranked_one[:rank_idx]])),
                      file=ranked_topic_word_file)

    @staticmethod
    def read_bow(file_path):
        """
        Read BOW file to run topic models with Variational Inference
        Do not modify this function
        :param file_path: The path of BOW file
        :return: documents list
        """
        split_pattern = re.compile(r'[ :]')
        docs = list()

        with open(file_path, 'r') as bow_file:
            for each_line in bow_file:
                split_line = split_pattern.split(each_line)

                word_ids = [int(x) for x in split_line[2::2]]
                word_counts = [int(x) for x in split_line[3::2]]

                cur_doc = LDAVI.Document(word_ids, word_counts)

                docs.append(cur_doc)

        return docs
