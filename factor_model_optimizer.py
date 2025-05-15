# coding: utf-8
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt, scipy as scp

import seaborn as sns, cvxpy as cp

svd = np.linalg.svd

class BetaLO(object):

    @staticmethod
    def get_betaLO_(lbda,beta,delta): # beta long-only
        sigmainv = Factor_Model.woodbury_fm(delta,beta,lbda)
        return lbda@beta.T@sigmainv@np.ones(len(beta))
    
    @staticmethod
    def get_betaLO__(lbda,beta,delta): # beta long-only
        return np.linalg.inv()

    def __init__(self,fmdict):
        self.__dict__.update(fmdict)

    def get_betaLO(self):
        return self.get_betaLO_(self.lbda,self.evecs,self.diagerrors)

class Optimizer(object):

    def __init__(self,fmdict,constraints):
        self.__dict__.update(fmdict)
        self.constraints = constraints

    def optimize(self):
        return self.get_betaLO_(self.lbda,self.evecs,self.diagerrors)


class Factor_Model(object):
       
        diagf = staticmethod(lambda diagm,f: np.diag(f(np.diag(diagm))))
        idiag = staticmethod(lambda diagm: Factor_Model.diagf(diagm, lambda e: 1/e))
        get_inv = staticmethod(lambda *args: [Factor_Model.idiag(e) for e in args])
        
        @staticmethod
        def subindex(*ars,idx,nums=[]):
            nums = nums + [None]*(len(ars)-len(nums))
            return [ar[np.ix_(*[idx for _ in range(num if num else len(ar.shape))])] for num,ar in zip(nums,ars)]
        
        @staticmethod
        def get_where(ar,round2=None,test = lambda x: x>0):
            '''
            Returns the indices of the array where the test is true, after rounding to round2 if given.
            If round2 is not given, the test is applied to the original array.
            '''
            ar_ = ar.copy()
            if round2 is not None: ar_ = ar_.round(round2)
            return np.where(test(ar_))[0]

        @staticmethod
        def woodbury_fm(A,beta,C): # computes inverse of Omega = A + UCV
            # specialized to C,A both diagonal for factor models,
            # C = diagonal factor covariance aka Lambda,
            # A = diagonal errors aka Delta, U = beta and V = beta.T,
            # so A + UCV = beta@Lambda@beta.T + Delta and the weights
            # = w \propto 
            # Omega^{-1}@1, where Omega^{-1} =
            # Ainv@(IdentityMatrix - beta@(Cinv + beta.T@Ainv@beta)^{-1}beta.T@Ainv)
            # with a denominator of 1^T@Omega^{-1}@1 (over 2 unless the quadratic term
            # equals 1/2w.T@Omega@w)
            Cinv,Ainv = Factor_Model.get_inv(C,A)
            return Ainv - Ainv@beta@np.linalg.inv(Cinv + beta.T@Ainv@beta)@beta.T@Ainv
       
        @classmethod
        def from_price(kls,price_series,annualize=False):
            return kls(kls.get_returns(price_series,annualize),annualize)
        
        def __init__(self,returns,annualize=False):
            self.returns = returns
            self.numtickers = len(returns.columns)
            self.cov = returns.cov()
            self.annualize = annualize

        def get_lowrank_sparse(self,eigenvector_indices,return_dict = False):
            '''
            Returns mapprox,diagonal lambda, lowrank = factor covariance,
            diagonal errors, eigenvalues, eigenvectors
            '''
            evals,evecs = scp.linalg.eigh(self.cov,subset_by_index = eigenvector_indices)
            lbda = np.diag(evals)
            vlbd = evecs@lbda
            lowrank = vlbd@evecs.T
            diagerrors = np.diag(np.diag(self.cov - lowrank))
            mapprox = pd.DataFrame(lowrank + diagerrors,index=self.cov.index,columns = self.cov.columns)
            return_values = (mapprox,lbda,lowrank,diagerrors,evals,evecs)
            return dict(zip('mapprox,lbda,lowrank,diagerrors,evals,evecs'.split(','),return_values)) if return_dict else return_values

class Minvar(object):
       
       @classmethod
       def from_fm(kls,n,fmdict,m=2):
           return kls(n,m,**{k1:fmdict[k] for k1,k in zip('lbda diag evecs'.split(),'lbda diagerrors evecs'.split())})

       def __init__(self,n=None,m=2,lbda=None,diag=None,evecs=None):
           self.n,self.m = n,m if evecs is None else evecs.shape
           self.w, self.f = [cp.Variable(i) for i in (n,m)]
           self.lbda,self.diag,self.F = lbda,diag,evecs
           if self.lbda is None and evecs: self.lbda = np.eye(evecs.shape[1])
           if self.diag is None: self.diag = np.eye(n)
           if self.F is None: self.F = np.ones((n,m))
           self.constraints = []
           # self.make_risk()
           
       def make_risk(self):
        #    self.risk = cp.quad_form(self.f,self.lbda) + cp.sum_squares(np.sqrt(self.diag)@self.w)
           self.risk = cp.quad_form(self.f,self.lbda) + cp.sum_squares(np.sqrt(self.diag)@self.w)

       def make_prob_factor(self):
           self.prob_factor = cp.Problem(
               cp.Maximize(-self.risk),
               self.constraints
           )

       def solve(self):
           self.make_prob_factor()
           self.prob_factor.solve()
           self.mv_index = self.get_mv_index()
           return self.w.value
           

       def update(self,update_dict):
        #    self.lbda,self.diag,self.F = lbda,diag,evecs
           self.__dict__.update(update_dict)
           self.make_risk()
           self.make_prob_factor()
           self.prob_factor.solve()
           self.mv_index = self.get_mv_index()
           return self.w.value

       def get_mv_index(self,tolerance = 1e-4):
           return np.where(self.w.value>tolerance)[0]
       
def subarray(idx, array):
    shp = array.shape
    print(shp)
    mx = max(shp)
    index_arrays = [widx[0] if k == mx else np.arange(k) for k in shp]
    print(index_arrays)
    midx = np.ix_(*index_arrays)#[:,:,0]
    print('midx',midx,'end midx')
    return array[midx]

def get_graph_params(df,round=None):
    fm= Factor_Model(df)
    twofactor_ = fm.get_lowrank_sparse((fm.numtickers-2,fm.numtickers-1),True)
    # onefactor_ = fm.get_lowrank_sparse([fm.numtickers-1,fm.numtickers-1],True)
    lbda,evecs,diagerrors = [twofactor_[k] for k in 'lbda evecs diagerrors'.split()]
    evecs = -evecs
    w = cp.Variable(evecs.shape[0])
    risk = cp.quad_form(w, evecs@lbda@evecs.T) +  cp.sum_squares(Factor_Model.diagf(diagerrors,np.sqrt)@w)
    prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w>=0])
    prob.solve()
    staridx = Factor_Model.get_where(w.value,5)
    # staridx = FM.get_where(w.value,round2=4,test=lambda x: x > 1e-3)
    # notstaridx = np.delete(np.arange(len(evecs)),staridx)
    # notstaridx = np.where(~(w.value.round(6)>0))[0]
    evecstar, diagstar = Factor_Model.subindex(evecs,diagerrors,idx=staridx,nums=[1])
    # derrstarinv,lbdainv = FM.get_inv(diagerrors,lbda)
    # evecstar = evecs[staridx]
    # diagstar = diagerrors[np.ix_(staridx,staridx)]
    betalo = BetaLO({'evecs':evecstar,'lbda':lbda,'diagerrors':diagstar})
    (blo := betalo.get_betaLO())
    return {'blo':blo,'staridx':staridx,'evecs':evecs,'diagerrors':diagerrors,'lbda':lbda,'w':w.value,'risk':risk.value,
            'evecstar':evecstar,'diagstar':diagstar}

def get_other(x,n): return (1 - x*n[0])/n[1] # n.dot((x,get_other(x))) =1 so with n as normal, (x,y) is on line n.dot(p) = 1

def make_graph(evecs,staridx,blo,portfolio_description='Specified'): # full evecs, staridx,
    # integer num is converted to a string=
    # num = '%i Stock' % evecs.shape[0] if num is None else num
    fig,ax = plt.subplots(1,1,figsize=(15,8))
    df = pd.DataFrame(evecs,columns="Eigenvector_2 Eigenvector_1".split()).reset_index(names='point')
    df = df.assign(LongOnly=df.point.apply(lambda p: 'In LO' if p in staridx else 'Not in LO'))
    sns.scatterplot(df,x = "Eigenvector_2", y = "Eigenvector_1", hue = 'LongOnly', ax = ax)
    
    xlims,ylims = [getattr(ax,k)() for k in ('get_'+s for s in 'xlim ylim'.split())]
    ys = [get_other(e,blo) for e in xlims]
    ax.plot(xlims,ys,c='red')
    ax.grid(True)
    ax.set_title(r'$\beta$s' +' in (Orange) and out (Blue) of MVLO' +' for %s Portfolio with Separating 2-Factor Line (Solid Red)\n   and Separating 1-Factor Line (Dotted Red)' % portfolio_description)
    # for row in model.iloc[:,:2].iterrows():
    #     ax.annotate(shorten_variable_name(tickerdict[row[0]]),row[1])
    ax.set_aspect('equal')
    return ax