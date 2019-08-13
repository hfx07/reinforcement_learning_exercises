# distutils: language = c++
# cython: language_level = 2, infer_types=True, boundscheck=False, wraparound=False, cdivision = True
from libc.math cimport sqrt, log, exp
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
import numpy

cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass random_device:
        ctypedef unsigned int result_type
        random_device() except +
        result_type operator()()
    cdef cppclass default_random_engine:
        ctypedef unsigned int result_type
        default_random_engine(result_type) except +
    cdef cppclass discrete_distribution[int]:
        ctypedef int result_type
        discrete_distribution() except +
        discrete_distribution(vector[double].iterator, vector[double].iterator) except +
        result_type operator()(default_random_engine&)
    cdef cppclass uniform_int_distribution[int]:
        ctypedef int result_type
        uniform_int_distribution() except +
        uniform_int_distribution(result_type, result_type) except +
        result_type operator()(default_random_engine&)
    cdef cppclass normal_distribution[double]:
        ctypedef double result_type
        normal_distribution(result_type, result_type) except +
        result_type operator()(default_random_engine&)

cdef extern from "<algorithm>" namespace "std" nogil:
    iterator max_element[iterator](iterator, iterator)
    void random_shuffle[iterator](iterator, iterator)
    
cdef extern from "<iterator>" namespace "std" nogil:
    #cdef cppclass iterator_traits[Iterator]:
    #    
    #iterator_traits[InputIterator]::difference_type distance(InputIterator first, InputIterator last)
    ptrdiff_t distance[iterator](iterator, iterator)

cdef int act_ucb(int levers, vector[double] &Q, vector[double] &Q_, vector[int] &N, int t, double c):
    cdef int i, a
    for i in range(levers):
        Q_[i] = Q[i] + c * sqrt(log(<double>t) / N[i])
    a = <int>distance(Q_.begin(), max_element(Q_.begin(), Q_.end()))
    return a

cdef int act_gradient(int levers, vector[double] &Q, vector[double] &pi, double sum_exp_Q, default_random_engine *generator):
    # Q is regarded as action preference H
    cdef int i, a
    for i in range(levers):
        pi[i] = exp(Q[i]) / sum_exp_Q
    cdef discrete_distribution[int] *discrete_dist_grad = new discrete_distribution[int](pi.begin(), pi.end())
    a = deref(discrete_dist_grad)(deref(generator))
    del discrete_dist_grad
    return a

cdef int act_epsilon_greedy(int levers, vector[double] &Q, double epsilon, discrete_distribution[int] *discrete_dist, uniform_int_distribution[int] *uniform_dist, default_random_engine *generator):
    cdef int i, a
    i = deref(discrete_dist)(deref(generator))
    if i == 0:
        a = deref(uniform_dist)(deref(generator))
    else:
        a = <int>distance(Q.begin(), max_element(Q.begin(), Q.end()))
    return a

cdef int act_greedy(int levers, vector[double] &Q):
    cdef int a
    a = <int>distance(Q.begin(), max_element(Q.begin(), Q.end()))
    return a

def run(int levers, double initQ, double [::1] q, int timeSteps, double epsilon, double stepSize, bint useSampleAverage, double c, bint useUCB, bint useGradient, bint withBaseLine):
    cdef random_device rd
    cdef default_random_engine *generator = new default_random_engine(rd())
    cdef normal_distribution[double] *normal_dist = new normal_distribution(0.0, 1.0)
    cdef vector[double] eps = [epsilon, 1.0 - epsilon]
    cdef discrete_distribution[int] *discrete_dist = new discrete_distribution(eps.begin(), eps.end())
    cdef uniform_int_distribution[int] *uniform_dist = new uniform_int_distribution(0, levers - 1)
    cdef int t, a, i
    # actions
    cdef vector[int] actions = [lv for lv in range(levers)]
    # initial estimates of action-values
    cdef vector[double] Q = [initQ for lv in range(levers)]
    cdef vector[double] Q_ = [initQ for lv in range(levers)]
    cdef vector[double] pi = [0.0 for lv in range(levers)]
    cdef double sum_exp_Q = 0.0
    # initial action counts
    cdef vector[int] N  = [0 for lv in range(levers)]
    # actions along time-steps
    A = numpy.zeros(timeSteps, dtype = numpy.intc)
    cdef int[::1] A_view = A
    # rewards along time-steps
    cdef double R_average = 0.0
    R = numpy.zeros(timeSteps)
    cdef double[::1] R_view = R
    if useUCB:
        random_shuffle(actions.begin(), actions.end())
        if useSampleAverage:
            for t in range(timeSteps):
                if t < levers:
                    a = actions[t]
                else:
                    a = act_ucb(levers, Q, Q_, N, t + 1, c)
                A_view[t] = a
                # all q(a) take independent random walks
                #for i in range(levers):
                #    q[i] = q[i] + 0.01 * deref(normal_dist)(deref(generator))
                R_view[t] = deref(normal_dist)(deref(generator)) + q[a] # ~ N(q, 1)
                # incremental implementation of sample averages
                N[a] = N[a] + 1
                Q[a] = Q[a] + (R_view[t] - Q[a]) / N[a]
        else:
            for t in range(timeSteps):
                if t < levers:
                    a = actions[t]
                else:
                    a = act_ucb(levers, Q, Q_, N, t + 1, c)
                A_view[t] = a
                # all q(a) take independent random walks
                #for i in range(levers):
                #    q[i] = q[i] + 0.01 * deref(normal_dist)(deref(generator))
                R_view[t] = deref(normal_dist)(deref(generator)) + q[a] # ~ N(q, 1)
                # incremental implementation of constant step-size
                N[a] = N[a] + 1
                Q[a] = Q[a] + (R_view[t] - Q[a]) * stepSize
    elif useGradient:
        for i in range(levers):
            sum_exp_Q += exp(Q[i])
        if withBaseLine:
            for t in range(timeSteps):
                a = act_gradient(levers, Q, pi, sum_exp_Q, generator)
                A_view[t] = a
                # all q(a) take independent random walks
                #for i in range(levers):
                #    q[i] = q[i] + 0.01 * deref(normal_dist)(deref(generator))
                R_view[t] = deref(normal_dist)(deref(generator)) + q[a] # ~ N(q, 1)
                N[a] = N[a] + 1
                # update action preference
                sum_exp_Q = 0.0
                for i in range(levers):
                    if i == a:
                        Q[i] = Q[i] + stepSize * (R_view[t] - R_average) * (1 - pi[i])
                    else:
                        Q[i] = Q[i] - stepSize * (R_view[t] - R_average) * pi[i]
                    sum_exp_Q += exp(Q[i])
                R_average = (R_average * t + R_view[t]) / (t + 1)
        else:
            for t in range(timeSteps):
                a = act_gradient(levers, Q, pi, sum_exp_Q, generator)
                A_view[t] = a
                # all q(a) take independent random walks
                #for i in range(levers):
                #    q[i] = q[i] + 0.01 * deref(normal_dist)(deref(generator))
                R_view[t] = deref(normal_dist)(deref(generator)) + q[a] # ~ N(q, 1)
                N[a] = N[a] + 1
                # update action preference
                sum_exp_Q = 0.0
                for i in range(levers):
                    if i == a:
                        Q[i] = Q[i] + stepSize * R_view[t] * (1 - pi[i])
                    else:
                        Q[i] = Q[i] - stepSize * R_view[t] * pi[i]
                    sum_exp_Q += exp(Q[i])
    elif epsilon > 0:
        if useSampleAverage:
            for t in range(timeSteps):
                a = act_epsilon_greedy(levers, Q, epsilon, discrete_dist, uniform_dist, generator)
                A_view[t] = a
                # all q(a) take independent random walks
                #for i in range(levers):
                #    q[i] = q[i] + 0.01 * deref(normal_dist)(deref(generator))
                R_view[t] = deref(normal_dist)(deref(generator)) + q[a] # ~ N(q, 1)
                # incremental implementation of sample averages
                N[a] = N[a] + 1
                Q[a] = Q[a] + (R_view[t] - Q[a]) / N[a]
        else:
            for t in range(timeSteps):
                a = act_epsilon_greedy(levers, Q, epsilon, discrete_dist, uniform_dist, generator)
                A_view[t] = a
                # all q(a) take independent random walks
                #for i in range(levers):
                #    q[i] = q[i] + 0.01 * deref(normal_dist)(deref(generator))
                R_view[t] = deref(normal_dist)(deref(generator)) + q[a] # ~ N(q, 1)
                # incremental implementation of constant step-size
                N[a] = N[a] + 1
                Q[a] = Q[a] + (R_view[t] - Q[a]) * stepSize
    else:
        if useSampleAverage:
            for t in range(timeSteps):
                a = act_greedy(levers, Q)
                A_view[t] = a
                # all q(a) take independent random walks
                #for i in range(levers):
                #    q[i] = q[i] + 0.01 * deref(normal_dist)(deref(generator))
                R_view[t] = deref(normal_dist)(deref(generator)) + q[a] # ~ N(q, 1)
                # incremental implementation of sample averages
                N[a] = N[a] + 1
                Q[a] = Q[a] + (R_view[t] - Q[a]) / N[a]
        else:
            for t in range(timeSteps):
                a = act_greedy(levers, Q)
                A_view[t] = a
                # all q(a) take independent random walks
                #for i in range(levers):
                #    q[i] = q[i] + 0.01 * deref(normal_dist)(deref(generator))
                R_view[t] = deref(normal_dist)(deref(generator)) + q[a] # ~ N(q, 1)
                # incremental implementation of constant step-size
                N[a] = N[a] + 1
                Q[a] = Q[a] + (R_view[t] - Q[a]) * stepSize
    del generator
    del normal_dist
    del discrete_dist
    del uniform_dist
    return A, R

#def test(runs, levers, timeSteps):
#    initQs_greedy = numpy.logspace(-2, 2, 5, base = 2)
#    average_rewards_greedy = numpy.zeros((5, timeSteps))
#    epsilons_greedy = numpy.logspace(-7, -2, 6, base = 2)
#    average_rewards_epsilon_greedy = numpy.zeros((6, timeSteps))
#    cs_ucb = numpy.logspace(-4, 2, 7, base = 2)
#    average_rewards_ucb = numpy.zeros((7, timeSteps))
#    alphas_gradient = numpy.logspace(-5, 2, 8, base = 2)
#    average_rewards_gradient = numpy.zeros((8, timeSteps))
#    for j in tqdm(range(runs)):
#        # true values of actions: start out equal
#        q = numpy.random.randn() * numpy.ones(levers)
#        for i, initQ in enumerate(initQs_greedy):
#            R = run(levers, initQ, q, timeSteps, 0, 0.1, None, None)
#            average_rewards_greedy[i, :] = average_rewards_greedy[i, :] + R
#        for i, epsilon in enumerate(epsilons_greedy):
#            R = run(levers, 0, q, timeSteps, epsilon, 0.1, None, None)
#            average_rewards_epsilon_greedy[i, :] = average_rewards_epsilon_greedy[i, :] + R
#        for i, c in enumerate(cs_ucb):
#            R = run(levers, 0, q, timeSteps, 'UCB', None, c, None)
#            average_rewards_ucb[i, :] = average_rewards_ucb[i, :] + R
#        for i, alpha in enumerate(alphas_gradient):
#            R = run(levers, 0, q, timeSteps, 'gradient', alpha, None, True)
#            average_rewards_gradient[i, :] = average_rewards_gradient[i, :] + R
#    average_rewards_greedy = average_rewards_greedy / runs
#    average_rewards_epsilon_greedy = average_rewards_epsilon_greedy / runs
#    average_rewards_ucb = average_rewards_ucb / runs
#    average_rewards_gradient = average_rewards_gradient / runs
#    fig = plt.figure()
#    plt.semilogx(initQs_greedy, average_rewards_greedy[:, -100000:].mean(axis = 1), basex = 2, color = 'k', label = r'greedy with optimistic initialization $\alpha = 0.1$')
#    plt.semilogx(epsilons_greedy, average_rewards_epsilon_greedy[:, -100000:].mean(axis = 1), basex = 2, color = 'r', label = r'$\epsilon$-greedy $\alpha = 0.1$')
#    plt.semilogx(cs_ucb, average_rewards_ucb[:, -100000:].mean(axis = 1), basex = 2, color = 'b', label = 'UCB')
#    plt.semilogx(alphas_gradient, average_rewards_gradient[:, -100000:].mean(axis = 1), basex = 2, color = 'g', label = 'gradient bandit')
#    plt.xlabel(r'$\epsilon$  $\alpha$  $c$  $Q_0$')
#    plt.ylabel('Average reward over last 100000 steps')
#    plt.legend()
#    fig.savefig('ex7_ten_armed_bandit_testbed_parameter_study_nonstationary.png')
#    plt.close('all')
#
#if __name__ == "__main__":
#    test(2000, 10, 200000)
