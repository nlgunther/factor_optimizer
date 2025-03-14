# coding: utf-8
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt,scipy as scp

import seaborn as sns, cvxpy as cp

svd = np.linalg.svd

class Factor_Model(object):

       @staticmethod
       def get_returns(price_series,annualize = False):
           returns = price_series.pct_change(fill_method=None).fillna(0)
           logreturns = (1 + returns).map(np.log).iloc[1:]
           if annualize: self.logreturns *= annualize
           return logreturns

       @staticmethod
       def idiag(d): return np.diag(1/np.diag(d))
       
       
       @staticmethod
       def get_inv(lbd,diag): 
           return [np.diag(1/np.diag(e)) for e in (lbd,diag)]
       
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

       def __init__(self,n,m=2,lbda=None,diag=None,evecs=None):
           self.n,self.m = n,m
           self.w, self.f = [cp.Variable(i) for i in (n,m)]
           self.lbda,self.diag,self.F = lbda,diag,evecs
           if self.lbda is None and evecs: self.lbda = np.eye(evecs.shape[1])
           if self.diag is None: self.diag = np.eye(n)
           if self.F is None: self.F = np.ones((n,m))
           self.constraints = []
           # self.make_risk()
           
       def make_risk(self):
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
           

       def update(self,lbda,diag,evecs):
           self.lbda,self.diag,self.F = lbda,diag,evecs
           self.make_risk()
           self.make_prob_factor()
           self.prob_factor.solve()
           self.mv_index = self.get_mv_index()
           return self.w.value

       def get_mv_index(self,tolerance = 1e-4):
           return np.where(self.w.value>tolerance)[0]
