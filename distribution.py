import numpy

from scipy.stats._distn_infrastructure import rv_discrete, _ShapeInfo
from scipy.stats._discrete_distns import _isintegral

class sis_gen(rv_discrete): # (S)tepwise (I)ncreasing (S)urvival distribution
    def _shape_info(self):
        # i: initial probability
        # f: final probability
        # c: convergence speed
        # M: maximum value
        return [_ShapeInfo("i", False, (0, 1), (True, False)),
                _ShapeInfo("f", False, (0, 1), (True, False)),
                _ShapeInfo("p", False, (0, 1), (True, True)),
                _ShapeInfo("M", True, (1, numpy.inf), (True, False))]
    
    def _evaluate_pmf(self, i, f, p, M):
        M = numpy.max(M)
        _p = i.copy()
        out = []
        remaining_mass = 1.0
        while len(out) <= M:
            out.append(remaining_mass * (1.0 - _p))
            remaining_mass -= out[-1]
            _p += (f - _p) * p
        return numpy.array(out) / numpy.sum(out, axis=0)
    
    def _evaluate_cdf(self, i, f, p, M):
        M = numpy.max(M)
        _p = i.copy()
        out = []
        cum_val = 0.0
        while len(out) <= M:
            out.append(cum_val + (1.0 - cum_val) * (1.0 - _p))
            cum_val = out[-1]
            _p += (f - _p) * p
        return numpy.array(out) / out[-1]
    
    def _rvs(self, i, f, p, M, size=None, random_state=None):
        rv_eval = random_state.uniform(0, 1, size=size)
        return self._ppf(rv_eval, i, f, p, M)
    
    def _argcheck(self, i, f, p, M):
        return (i >= 0) & (i < 1) & (f >= 0) & (f < 1) & (p >= 0) & (p <= 1.0) & _isintegral(M) & (M >= 1)
    
    def _get_support(self, i, f, p, M):
        return self.a, M
    
    def _pmf(self, x, i, f, p, M):
        evaluated = self._evaluate_pmf(i, f, p, M)
        out = numpy.NaN * numpy.ones_like(x)
        mask = _isintegral(x) & (x <= M)
        out[mask] = evaluated[x[mask].astype(int), numpy.nonzero(mask)[0]]
        return out
        
    def _cdf(self, x, i, f, p, M):
        evaluated = self._evaluate_cdf(i, f, p, M)
        out = numpy.NaN * numpy.ones_like(x)
        out = evaluated[numpy.minimum(x, M).astype(int),
                        numpy.arange(evaluated.shape[1], dtype=int)]
        return out
    
    def _ppf(self, x, i, f, p, M):
        cdf_eval = self._evaluate_cdf(i, f, p, M)
        if cdf_eval.ndim > 1:
            assert cdf_eval.shape[1] == len(x)
            res = [numpy.ceil(numpy.interp(_x, _cdf, numpy.arange(len(_cdf)), left=0, right=_M))
                   for _x, _cdf, _M in zip(x, cdf_eval.transpose(), M)]
            return numpy.array(res)
        return numpy.ceil(numpy.interp(x, cdf_eval, 
                          numpy.arange(len(cdf_eval)),
                                       left=0, right=M)).astype(int)
    
    def _stats(self, i, f, p, M, moments='mv'):
        evaluated = self._evaluate_pmf(i, f, p, M)
        x = numpy.arange(len(evaluated)).reshape((-1, 1))
        mu = (evaluated * x).sum(axis=0)
        var = (evaluated * ((x - mu) ** 2)).sum(axis=0)
        if 's' in moments or 'k' in moments:
            raise NotImplemented()
        return mu, var, None, None
    
    def pid_curve(self, i, f, p, M):
        x = numpy.arange(M + 1)
        y = []
        _p = i
        while len(y) < len(x):
            y.append(_p)
            _p = _p + (f - _p) * p
        return x, numpy.array(y)
    
sis = sis_gen(name="sis")


def cut_zeros_and_shift(distr_in):
    # assert not numpy.any([hasattr(_a, "__len__") for _a in distr_in.args])
    p_zero = 1.0 - distr_in.args[0]
    i_new = distr_in.args[0] + (distr_in.args[1] - distr_in.args[0]) * distr_in.args[2]
    return sis(i_new, distr_in.args[1], distr_in.args[2], distr_in.args[3] - 1), p_zero

def add_zeros_and_shift(distr_in):
    # assert not numpy.any([hasattr(_a, "__len__") for _a in distr_in.args])
    p_1 = distr_in.args[0]
    a1 = distr_in.args[1]
    a2 = distr_in.args[2]
    p_0 = (p_1 - a1*a2) / (1-a2)
    return sis(p_0, a1, a2, distr_in.args[3] + 1)
    
