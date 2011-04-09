"""
(C) Mathieu Blondel - 2010

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""

import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.int
ctypedef np.int_t DTYPE_t
from scipy.special import gammaln

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(np.ndarray[np.double_t, ndim=1] vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    li = []
    cdef int idx, i
    for idx in vec.nonzero()[0]:
        for i in range(int(vec[idx])):
            li.append(idx)
    return li

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class LdaSampler(object):

    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _conditional_distribution(self, int m, int w, 
                                   np.ndarray[DTYPE_t, ndim=2] nzw,
                                   np.ndarray[DTYPE_t, ndim=2] nmz,
                                   np.ndarray[DTYPE_t, ndim=1] nz,
                                   np.ndarray[DTYPE_t, ndim=1] nm):
        """
        Conditional distribution (vector of size n_topics).
        """
        cdef int vocab_size = nzw.shape[1]
        cdef int z, n_topics
        n_topics = self.n_topics
        cdef double beta = self.beta
        cdef double alpha = self.alpha
        cdef np.ndarray[np.double_t, ndim=1] p_z
        p_z = np.zeros(n_topics, dtype=np.double)
        for z in xrange(n_topics):
            p_z[z] = (nzw[z, w] + beta) * (nmz[m, z] + alpha)/ \
                (nz[z] + beta * vocab_size)
        # normalize to obtain probabilities
        cdef double p_z_sum  = 0
        for z in xrange(n_topics):
            p_z_sum += p_z[z]

        #solve conversion rounding error
        cdef double partial_sum = 0
        for z in xrange(n_topics - 1):
            p_z[z] /= p_z_sum
            partial_sum += p_z[z]
        p_z[n_topics-1] = 1.0 - partial_sum
        return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        cdef int vocab_size = self.nzw.shape[1]
        cdef int n_docs = self.nmz.shape[0]
        cdef int lik = 0

        cdef int z, w
        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        cdef int V = self.nzw.shape[1]
        cdef np.ndarray[np.double_t, ndim = 2] num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def run(self, np.ndarray[np.double_t, ndim=2] matrix, int maxiter=30):
        """
        Run the Gibbs sampler.
        """
        cdef int n_docs, vocab_size
        #unboxing behaviour not supported
        n_docs = matrix.shape[0]
        vocab_size = matrix.shape[1]

        # initialize
        cdef int it, m, i, w, z
        # number of times document m and topic z co-occur
        cdef np.ndarray[DTYPE_t, ndim=2] nmz = np.zeros((n_docs, self.n_topics), dtype=DTYPE)
        # number of times topic z and word w co-occur
        cdef np.ndarray[DTYPE_t, ndim=2] nzw = np.zeros((self.n_topics, vocab_size), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] nm = np.zeros(n_docs, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] nz = np.zeros(self.n_topics, dtype=DTYPE)

        n_docs = matrix.shape[0] 
        vocab_size = matrix.shape[1]

        self.topics = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                nmz[m,z] += 1
                nm[m] += 1
                nzw[z,w] += 1
                nz[z] += 1
                self.topics[(m,i)] = z

        li = []
        cdef np.ndarray[np.double_t, ndim=1] p_z 

        for it in range(maxiter):
            for m in range(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m,i)]
                    nmz[m,z] -= 1
                    nm[m] -= 1
                    nzw[z,w] -= 1
                    nz[z] -= 1

                    p_z = self._conditional_distribution(m, w, nzw, nmz, nz, nm)
                    z = sample_index(p_z)

                    nmz[m,z] += 1
                    nm[m] += 1
                    nzw[z,w] += 1
                    nz[z] += 1
                    self.topics[(m,i)] = z
            self.nzw = nzw
            self.nmz = nmz

            # FIXME: burn-in and lag!
            li.append(self.phi())

        return li

