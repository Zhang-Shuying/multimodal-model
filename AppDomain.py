'''
Creator:     Kael (Zhongyu Wang)
Email:       wzy.kael@gmail.com
Time:        Jun. 21st, 2019
Updated:     Dec. 22nd, 2020
Description:
A revolutional version for dataObj
The most strict applicability domain - non-parameter KDE based AD has been introduced in this version
ABBR. used
L: list, T: tuple, S: set, df: pandas.DataFrame, sr: pandas.Series
'''
# Jun 18th, 2019
import pandas as pd
import numpy as np
from scipy import interp
from scipy.optimize import linprog # for convex hull test
import pickle # pickle.dump(obj, file, protocol=None)
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
#
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, DistanceMetric
from sklearn.metrics.pairwise import pairwise_kernels
#
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, balanced_accuracy_score, precision_score, recall_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

"""
Author: Kael
Date: 2020-10-16
Compact module for generating "network-like similarity graphs (NSG)"
Node properties (e.g., color) could be specified, so that the NSG could be
a representation of the structure-activity landscape (SAL) or the structure-predicted error landscape
Node size could also be adjusted, but generally the local discontinuity score (LDS) would be represented by the size.
"""

# the new applicability domain that integrate network-like similarity graphs
# of the structure-activity landscape, and local discontinuity analysis 
# taking use of AppDomainFpSimilarity
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout # if graphviz is unavailable, local layout in NX could be alternatives
from networkx.algorithms.community import greedy_modularity_communities, modularity
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
# here is some parameters that could be changed by the users in the upper level scripts
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['xtick.labelsize'] = 9.5
rcParams['ytick.labelsize'] = 9.5
'''
# Draw molecules in 2D, 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
'''
# A quick MCS search is allowed.
from rdkit.Chem import rdFMCS

# confine the k-clique in its convex hull:
from scipy.spatial import ConvexHull

def sigmoidWt(x, sCutoff, stiffness=15):
    return 1/(1 + np.exp(-stiffness*(x-sCutoff)))  
#
def rigidWt(x, sCutoff):
    y = np.ones(shape=x.shape)   
    y[x < sCutoff] = 0
    return y
#
def leakyWt(x, sCutoff, baselineWt=0, slope=1):
    y = baselineWt*np.ones(shape=x.shape) + slope*(x-sCutoff)  
    y[x < sCutoff] = 0
    return y
#
def expWt(x, a=18, eps=1e-6):    
    return np.exp(-a*(1-x)/(x + eps))
#
def complexWt(x, sCutoff, baselineWt=1, slope=2, pw=2, eps=1e-6):
    '''
    if rigidWt is needed, use kwarg={'sCutoff':sCutoff2,'baselineWt':1.0,'slope':0,'pw':0,'eps':0}
    #
    if only power function is used, the following is somehow good.
    kwarg={'sCutoff':0.0,'baselineWt':0.0,'slope':15/(1-sCutoff2),'pw':5,'eps':0}
    '''
    y = baselineWt*np.ones(shape=x.shape) + slope*(x-sCutoff)**pw
    y[x < sCutoff] = eps
    return y
#
def visualizeWtScheme(wtFunc, kwarg, interpN=1000):
    xt = np.arange(0,1,1/interpN)
    yt = wtFunc(xt, **kwarg)
    plt.plot(xt,yt,'r-')
#

class NSG:
    def __init__(self, df_train, yCol, smiCol=None, XCol=None, cmpdIdx=None):
        '''df_train: pandas dataframe containing training-set compounds
        yCol: y value that to be predicted
        smiCol: dataframe column for SMILES
        XCol: dataframe columns for descriptors, should be a sequence like list or pandas index
        cmpdIdx=None: specify the index/key for compounds
        by default, index of the df_train should be key for the compounds
        '''
        if (smiCol is None) and (XCol is None):
            raise ValueError('Either smiCol or XCol has to be provided!')
        self.df_train = df_train[[yCol]]
        self.yCol = yCol
        if smiCol is not None:
            self.df_train = self.df_train.join(df_train[[smiCol]])
            self.smiCol = smiCol
        if XCol is not None:
            self.df_train = self.df_train.join(df_train[XCol])
            self.X = df_train[XCol].values
            self.XCol = XCol
        if cmpdIdx is not None:
            self.df_train.set_index(cmpdIdx, inplace=True)
        if len(df_train.index.unique()) != df_train.index.shape[0]:
            raise ValueError('Redundant cmpdIdx must be removed!')
        self._calcYDiffMatrix() # calculate the (signed) y difference matrix
        #
    def _calcYDiffMatrix(self):
        eucD = DistanceMetric.get_metric('euclidean')
        self.dfEDM = pd.DataFrame(eucD.pairwise(self.df_train[[self.yCol]]), index=self.df_train.index, columns=self.df_train.index)
        a = self.df_train[self.yCol].values
        self.dfSDM = pd.DataFrame(np.subtract.outer(a,a), index=self.df_train.index, columns=self.df_train.index)
    def calcPairwiseSimilarityWithFp(self, fpType, **kw):
        # Morgan(bit), radius=2, nBits=1024
        # MACCS_keys
        self.kw = kw
        self.fpType = fpType
        if fpType == 'Morgan(bit)':
            self.fpTypeStr = 'Morgan(r{:d}n{:d})'.format(kw['radius'],kw['nBits'])
        else:
            self.fpTypeStr = fpType
        self.adfs = AppDomainFpSimilarity(self.df_train, smiCol=self.smiCol)
        self.adfs.fpSimilarity_analyze(fpType, **kw)
        # pairwise similarity matrix, PSM
        self.dfPSM = pd.DataFrame(self.adfs.SM_trained, index=self.df_train.index, columns=self.df_train.index)
        self._calcSlopeMatrix()
        idxPair = np.triu_indices(self.dfPSM.index.shape[0],1)
        self.df_pSSN = pd.DataFrame({'A': self.dfPSM.index[idxPair[0]],'B':   self.dfPSM.index[idxPair[1]],
                                     'Tc':self.dfPSM.values[idxPair],  'diff':self.dfSDM.values[idxPair]})
    def calcPairwiseSimilarityWithBoolean(self, fpTypeStr='x',metric='jaccard'):
        # Morgan(bit), radius=2, nBits=1024
        # MACCS_keys
        self.fpTypeStr = fpTypeStr+'.'+metric
        jaccardDistPair = DistanceMetric.get_metric(metric)
        # pairwise similarity matrix, PSM
        self.dfPSM = pd.DataFrame(1-jaccardDistPair.pairwise(self.X), index=self.df_train.index, columns=self.df_train.index)
        self._calcSlopeMatrix()
        idxPair = np.triu_indices(self.dfPSM.index.shape[0],1)
        self.df_pSSN = pd.DataFrame({'A': self.dfPSM.index[idxPair[0]],'B':   self.dfPSM.index[idxPair[1]],
                                     'Tc':self.dfPSM.values[idxPair],  'diff':self.dfSDM.values[idxPair]})
    def calcPairwiseKernelWithBoolean(self, fpTypeStr='kernel',metric='rbf',**kwds):
        # pairwise kernel is used as pairwise similarity
        self.fpTypeStr = fpTypeStr+'.'+metric
         # pairwise kernel (similarity) matrix, PSM
        self.kernelKeywords = kwds
        self.dfPSM = pd.DataFrame(pairwise_kernels(self.X, metric=metric, **kwds), index=self.df_train.index, columns=self.df_train.index)
        self._calcSlopeMatrix()
        idxPair = np.triu_indices(self.dfPSM.index.shape[0],1)
        self.df_pSSN = pd.DataFrame({'A': self.dfPSM.index[idxPair[0]],'B':   self.dfPSM.index[idxPair[1]],
                                     'Tc':self.dfPSM.values[idxPair],  'diff':self.dfSDM.values[idxPair]})
    def loadPairwiseSimilarity(self, sMatrix, fpTypeStr='loaded'):
        self.fpTypeStr = fpTypeStr
        self.dfPSM = pd.DataFrame(sMatrix, index=self.df_train.index, columns=self.df_train.index)
        self._calcSlopeMatrix()
        idxPair = np.triu_indices(self.dfPSM.index.shape[0],1)
        self.df_pSSN = pd.DataFrame({'A': self.dfPSM.index[idxPair[0]],'B':   self.dfPSM.index[idxPair[1]],
                                     'Tc':self.dfPSM.values[idxPair],  'diff':self.dfSDM.values[idxPair]})
    def genNSG(self, sCutoff=0.55, includeSingleton=False):
        # generate an NSG with specified similarity cutoff (sCutoff)
        # singletons which are not connected with other nodes are by default to be ignored
        #
        self.GsCutoff = sCutoff
        if 'LD|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff) not in self.df_train.columns:
            self.calcLocalDiscontinuityScore(sCutoff)
        df_pSSN0 = self.df_pSSN[self.df_pSSN.Tc >= sCutoff]
        self.G = nx.from_pandas_edgelist(df_pSSN0,'A','B','Tc')
        if includeSingleton: self.G.add_nodes_from(self.df_train.index)
        self.genGss()
    def _calcSlopeMatrix(self):
        self.dfSlopeM = pd.DataFrame(self.dfEDM.values*self.dfPSM.values, index=self.dfPSM.index, columns=self.dfPSM.columns)
        self.dfSignedSlopeM = pd.DataFrame(self.dfSDM.values*self.dfPSM.values, index=self.dfPSM.index, columns=self.dfPSM.columns)
    def calcLocalDiscontinuityScore(self, sCutoff=0.55):
        dfMask = (self.dfPSM >= sCutoff)
        # unsigned section
        dfSlp = self.dfSlopeM[dfMask]
        dfCliff = self.dfEDM[dfMask]
        srDg = dfMask.sum(axis=1) - 1
        srDg.name = 'degree|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        srDens = self.dfPSM[dfMask].sum(axis=1) - 1
        srDens.name = 'density|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        srLd = dfSlp.sum(axis=1) / srDg
        srLd.name = 'LD|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        srGrad = dfSlp.sum(axis=1) / srDens
        srGrad.name = 'Gradient|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        srMaxSlp = dfSlp.max(axis=1)
        srMaxSlp.name = 'maxSlope|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        srMaxDiff = dfCliff.max(axis=1)
        srMaxDiff.name = 'maxDiff|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        # signed section
        dfSignedSlp = self.dfSignedSlopeM[dfMask]
        srSignedLd = dfSignedSlp.sum(axis=1) / srDg
        srSignedLd.name = 'signedLD|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        srSignedGrad = dfSignedSlp.sum(axis=1) / srDens
        srSignedGrad.name = 'signedGradient|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        # for classification task
        srRatio = dfCliff.sum(axis=1) / srDg
        srGini = srRatio * (1 - srRatio)
        srGini.name = 'Gini|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)
        # merge info
        self.df_train = self.df_train.join(srDg).join(srDens).join(srLd).join(srGrad).join(srSignedLd).join(srSignedGrad).join(srMaxSlp).join(srMaxDiff).join(srGini)
    def calcWtLocalDiscontinuityScore(self, wtFunc=None, kwarg={'sCutoff':0.55,'stiffness':15}, plotWtFunc=False):
        if wtFunc is None:
            wtFunc = sigmoidWt
        if plotWtFunc:
            visualizeWtScheme(wtFunc, kwarg)
        dfWt = pd.DataFrame(wtFunc(self.dfPSM.values, **kwarg), index=self.dfPSM.index, columns=self.dfPSM.columns)
        dfWPSM = self.dfPSM * dfWt
        srWtDg = dfWt.sum(axis=1) - np.diag(dfWt)
        srWtDg.name = 'wtDegree|{:s}'.format(self.fpTypeStr)
        dfWtSlopeM = pd.DataFrame(self.dfEDM.values*dfWPSM.values, index=self.dfPSM.index, columns=self.dfPSM.columns)
        dfWtSignedSlopeM = pd.DataFrame(self.dfSDM.values*dfWPSM.values, index=self.dfPSM.index, columns=self.dfPSM.columns)
        srWtLd = dfWtSlopeM.sum(axis=1) / srWtDg
        srWtLd.name = 'wtLD|{:s}'.format(self.fpTypeStr)
        srWtSignedLd = dfWtSignedSlopeM.sum(axis=1) / srWtDg
        srWtSignedLd.name = 'wtSignedLD|{:s}'.format(self.fpTypeStr)
        return self.df_train[[self.smiCol,self.yCol]].join(srWtDg).join(srWtLd).join(srWtSignedLd)
    
    
    def calcWtLocalDiscontinuityScore1(self, wtFunc=None, kwarg={'sCutoff':0.55,'stiffness':15}, plotWtFunc=False):
        if wtFunc is None:
            wtFunc = sigmoidWt
        if plotWtFunc:
            visualizeWtScheme(wtFunc, kwarg)        
        dfWt = pd.DataFrame(wtFunc(self.dfPSM.values), index=self.dfPSM.index, columns=self.dfPSM.columns)
        dfWPSM = self.dfPSM * dfWt
        srWtDg = dfWt.sum(axis=1) - np.diag(dfWt)
        srWtDg.name = 'wtDegree|{:s}'.format(self.fpTypeStr)
        srWtDens = dfWPSM.sum(axis=1) - np.diag(dfWPSM)
        srWtDens.name = 'wtDensity|{:s}'.format(self.fpTypeStr)
        dfWtSlopeM = pd.DataFrame(self.dfEDM.values*dfWPSM.values, index=self.dfPSM.index, columns=self.dfPSM.columns)
        dfWtSignedSlopeM = pd.DataFrame(self.dfSDM.values*dfWPSM.values, index=self.dfPSM.index, columns=self.dfPSM.columns)
        srWtLd = dfWtSlopeM.sum(axis=1) / srWtDg
        srWtLd.name = 'wtLD|{:s}'.format(self.fpTypeStr)
        srWtSignedLd = dfWtSignedSlopeM.sum(axis=1) / srWtDg
        srWtSignedLd.name = 'wtSignedLD|{:s}'.format(self.fpTypeStr)
        return self.df_train[[self.smiCol,self.yCol]].join(srWtDg).join(srWtDens).join(srWtLd).join(srWtSignedLd)
    
    
    
    def genQTSM(self, dfQuery, smiCol):
        '''generate query-training compounds similarity matrix
        dfQuery: dataframe that contains the 'featuresChosen' with training set
        # query-training similarity matrix'''
        return pd.DataFrame(self.adfs.fpSimilarity_xenoCheck(dfQuery, smiCol), index=dfQuery.index, columns=self.df_train.index)
    def queryCliffInAD(self, dfQTSM, sCutoff=0.55):
        '''all fingerprint type compatible xenoFilter
        dfQTSM: the Tanimoto similarity matrix returned by AppDomainFpSimilarity.fpMorganTc_xenoCheck()
        sCutoff(default=0.25): by (>=) which structures 'similar' to training set are identified;
                                    when the similarity is larger, the AD will be stricter.
        '''
        if 'LD|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff) not in self.df_train.columns:
            self.calcLocalDiscontinuityScore(sCutoff)
        dfMask = dfQTSM >= sCutoff
        dfInAD = pd.DataFrame()
        for qCmpd in dfQTSM.index:
            tri = dfMask.loc[qCmpd]
            nSimi = tri.sum()
            if nSimi > 0: # only compounds with nSimi > 0 is considered in AD
                dfInAD.loc[qCmpd,'nSimi|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = nSimi
                srMaxSlp = self.df_train.loc[tri]['maxSlope|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)]
                srMaxDiff = self.df_train.loc[tri]['maxDiff|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)]
                dfInAD.loc[qCmpd,'maxDiff|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = srMaxDiff.max()
                dfInAD.loc[qCmpd,'meanDiff|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = srMaxDiff.mean()
                dfInAD.loc[qCmpd,'maxSlope|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = srMaxSlp.max()
                dfInAD.loc[qCmpd,'meanSlope|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = srMaxSlp.mean()
        return dfInAD
    def queryLDInAD(self, dfQTSM, sCutoff):
        '''all fingerprint type compatible xenoFilter
        dfQTSM: the Tanimoto similarity matrix returned by AppDomainFpSimilarity.fpMorganTc_xenoCheck()
        sCutoff: by (>=) which structures 'similar' to training set are identified;
                                    when the similarity is larger, the AD will be stricter.
        '''
        if 'LD|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff) not in self.df_train.columns:
            self.calcLocalDiscontinuityScore(sCutoff)
        dfMask = dfQTSM >= sCutoff
        dfInAD = pd.DataFrame()
        for qCmpd in dfQTSM.index:
            tri = dfMask.loc[qCmpd]
            nSimi = tri.sum()
            if nSimi > 0: # only compounds with nSimi > 0 is considered in AD
                dfInAD.loc[qCmpd,'nSimi|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = nSimi
                srLd = self.df_train.loc[tri]['LD|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)]
                srDg = self.df_train.loc[tri]['degree|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)]
                dfInAD.loc[qCmpd,'meanNSimiDegree|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = srDg.mean()
                dfInAD.loc[qCmpd,'meanLD|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = srLd.mean()
                dfInAD.loc[qCmpd,'weightedLD|{:s}|{:.2f}'.format(self.fpTypeStr, sCutoff)] = (srLd*srDg).sum()/srDg.sum()
        return dfInAD
    def queryADMetrics(self, dfQTSM, dfWtLD=None, wtFunc1=expWt, kw1={'a':15}, wtFunc2=expWt, kw2={'a':15}, code=''):
        '''all fingerprint type compatible xenoFilter
        dfQTSM:    the Tanimoto similarity matrix returned by AppDomainFpSimilarity.fpMorganTc_xenoCheck()
        dfWtLD:    dataframe of the 'wt info' of the SAL, generated by NSG.calcWtLocalDiscontinuityScore
        '''
        if dfWtLD is None:
            dfWtLD = self.calcWtLocalDiscontinuityScore(wtFunc1, kw1)
        dfSimiWt = pd.DataFrame(wtFunc2(dfQTSM, **kw2), index=dfQTSM.index, columns=dfQTSM.columns)
        LDw = dfWtLD.loc[:,'wtLD|{:s}'.format(self.fpTypeStr)]
        mask = ~LDw.isin([np.nan, np.inf]) # NaN or infinite values cannot be considered
        simiDens = dfSimiWt.loc[:,mask].sum(axis=1) # exclude contribution from those not considered
        simiDens.name = 'simiDensity'+code
        wtLDw = dfSimiWt.loc[:,mask].dot(LDw[mask])/ simiDens
        wtLDw.name = 'simiWtLD_w'+code
        Dgw = dfWtLD['wtDegree|{:s}'.format(self.fpTypeStr)]
        wtDgw = dfSimiWt.loc[:,mask].dot(Dgw[mask]) / simiDens
        wtDgw.name = 'simiWtDegee_w'+code
        return pd.DataFrame(simiDens).join(wtDgw).join(wtLDw)
    def genGss(self, showNodes=None):
        if showNodes is None:
            self.Gss = self.G
        else:
            self.Gss = self.G.subgraph(showNodes)
    def calcCliques(self):
        self.cliqueL = list(nx.find_cliques(self.Gss))
        self.cliqueSize = np.array([len(clique) for clique in self.cliqueL])
        self.sortedIdx = self.cliqueSize.argsort()[::-1]
    def keepCliques(self, minCliqueSize=4, maxISetSize=0):
        if self.cliqueSize.max() < minCliqueSize:
            return []
        kCliquesKept = [self.cliqueL[self.sortedIdx[0]]]
        cUSet = set(self.cliqueL[self.sortedIdx[0]])
        for idx in sortedIdx[1:]:
            cTmpSet = set(cliqueL[idx])
            if len(cTmpSet) < minCliqueSize:
                break
            #elif len(cUSet.intersection(cTmpSet))/len(cTmpSet) > 0.1: continue
            elif len(cUSet.intersection(cTmpSet)) <= maxISetSize:
                cUSet = cUSet.union(cTmpSet)
                kCliquesKept.append(cliqueL[idx])
            else: continue
        return kCliquesKept
    def calcComm(self):
        self.commList = list(greedy_modularity_communities(self.Gss))
        self.commSizeArr = np.array([len(comm) for comm in self.commList])
        # update community tab in the self.df_train
        self.df_train['comm|{:s}|{:.2f}'.format(self.fpTypeStr, self.GsCutoff)] = 'singleton'
        for i, comm in enumerate(self.commList):
            self.df_train.loc[comm,'comm|{:s}|{:.2f}'.format(self.fpTypeStr, self.GsCutoff)] = 'comm#{:d}'.format(i+1)
    def filterComm(self, minCommSize=3):
        if self.commSizeArr.max() < minCommSize: return []
        uList = []
        for idx, size in enumerate(self.commSizeArr):
            if self.commSizeArr[idx] >= minCommSize:
                uList.append(self.commList[idx])
        return uList
    def calcCC(self):
        '''connected components in the network'''
        self.ccL = list(nx.connected_components(self.Gss))
        self.ccSizeArr = np.array([len(cc) for cc in self.ccL])
        # update CC tab in the self.df_train
        self.df_train['CC|{:s}|{:.2f}'.format(self.fpTypeStr, self.GsCutoff)] = 'singleton'
        for i, cc in enumerate(self.ccL):
            self.df_train.loc[cc,'CC|{:s}|{:.2f}'.format(self.fpTypeStr, self.GsCutoff)] = 'CC#{:d}'.format(i+1)
    def filterCCbySize(self, sizeLower=0, sizeUpper=10000):
        uList = []
        for idx, size in enumerate(self.ccSizeArr):
            if (self.ccSizeArr[idx] >= sizeLower) & (self.ccSizeArr[idx] <= sizeUpper):
                uList.extend(self.ccL[idx])
        return uList
    def filterCCwithNodes(self, nodes=[]):
        uList = []
        ccIds = self.df_train.loc[nodes,'CC|{:s}|{:.2f}'.format(self.fpTypeStr, self.GsCutoff)]
        uList.extend(ccIds[ccIds == 'singleton'])
        for ccId in ccIds[ccIds != 'singleton'].unique():
            uList.extend(self.ccL[int(ccId[3:]) - 1])
        return uList
    def neighborhoodPlot(self, nbDiff=2, eps=1e-5, plot=True):
        '''
        Patterson neighborhood behavior plot.
        nbDiff=2: neighborhood difference used for calculating neighborhood radius (may fail for similarity in [0, 1])
        eps: used for avoiding the zero divider.
        x axis: pairwise distance or "1 - similarity + eps"
        y axis: pairwise absolute difference of activity
        red line & dot: optimal diagonal and its determinator
        #
        Ref.: Patterson DE, et al. Neighborhood behavior: a useful concept for validation of "molecular diversity" descriptors
        '''
        x = 1 + eps - self.df_pSSN['Tc'].values
        y = np.abs(self.df_pSSN['diff'].values)
        #
        xmax = x.max()
        ymax = y.max()
        k = y/x
        maskU = k > ymax/xmax
        print(maskU.sum())
        xu = x[maskU]
        yu = y[maskU]
        dens = []
        for x0,y0 in zip(xu,yu):
            Atri0 = 0.5 * x0 * ymax
            k0 = y0 / x0
            N0 = ((x < x0)&(k <= k0)).sum() # have to be "x < x0", if x <= x0, error may arise!
            dens0 = N0 / Atri0
            dens.append(dens0)
        densArr = np.array(dens)
        mask = np.arange(0, densArr.shape[0])
        idxmax = mask[densArr == densArr.max()]
        self.NB_slope = yu[idxmax] / xu[idxmax]
        densEntire = x.shape[0]/ (xmax * ymax)
        densLRT = (k <= self.NB_slope).sum() / ((xmax + xmax-xu[idxmax])*ymax*0.5)
        self.NB_enhancement = densLRT / densEntire
        self.NB_radius = nbDiff/self.NB_slope
        if plot:
            xt = np.arange(0,xmax,0.01)
            yt = xt * self.NB_slope
            plt.plot(x, y, '.', alpha=0.01)
            plt.plot(xt, yt, 'r')
            plt.plot([xu[idxmax]],[yu[idxmax]],'ro',)
    def SALI_analyze(self):
        # maybe I should add this, with the SALI curve. Guha et al., 2008a,b
        pass
#







import matplotlib
from pyvis.network import Network
class NSGVisualizer:
    def __init__(self, nsg, weightName='Tc'):
        self.nsg = nsg
        try:
            Gss = nsg.Gss
        except:
            nsg.Gss = nsg.G
        # get weights of edges as numpy array
        self.eWtArr = np.array([d[weightName] for u,v,d in nsg.Gss.edges(data=True)])
    def calcPos(self, prog='neato', picklePos=True, pickledPosPrefix=''):
        #
        self.pos = graphviz_layout(self.nsg.Gss, prog=prog)
        if picklePos:
            pickledPosName = pickledPosPrefix+'pos_{:s}_{:.2f}.pickle'.format(self.nsg.fpTypeStr,self.nsg.GsCutoff)
            with open(pickledPosName, 'wb') as f:
                pickle.dump(self.pos, f)
    def loadPos(self, posPickleFilePath):
        with open(posPickleFilePath, 'rb') as f:
            self.pos = pickle.load(f)
    def render(self, nodeSizeCol, nodeColorCol, nodeEdgeColors='gray', nodeEdgeWeights=0.0, drawNodeLabels=False, nodesShown=None, font_size=8, 
    markComm=True, minCommSize=3, commEdgeColor='k', commAlpha=0.1, commEdgeWidth=0.5, commEdgePad=20, commEdgeInterpN=36, ratioAdjust=None,
    figsizeTup = (6.3,3.576), legendWidth=0.7, drawEdges=True, edgeAlpha=0.25, vmin=None, vmax=None, isContinuousValue=True, nLegend_activity=1,
    leftPadRatio=0.05, rightPadRatio=0.05, bottomPadRatio=0.05, topPadRatio=0.05, annotateNodePos=dict(), annotatePosStyle='offset points',
    sizeBase=0.05, sizeScale=30, cmapName = 'RdYlGn_r', nLegend_LD=4, bboxTup=(0.25,0.5,0.25,0.8),
    showLegend=True, savePng=True, PngNamePrefix='', groupsToMark=[]):
        if ratioAdjust == None:
            ratioAdjust = figsizeTup[1]/(figsizeTup[0]-legendWidth)
        df_smi = self.nsg.df_train
        Gss = self.nsg.Gss
        # pos and pretty
        pos = self.pos
        posArr = np.array(list(pos.values()))
        xmax, ymax = posArr.max(axis=0)
        xmin, ymin = posArr.min(axis=0)
        leftW, rightW = xmin-leftPadRatio*(xmax-xmin), xmax+rightPadRatio*(xmax-xmin)
        bottomW, topW = ymin-bottomPadRatio*(ymax-ymin), ymax+topPadRatio*(ymax-ymin)
        #
        eWtArr = self.eWtArr
        cm = plt.get_cmap(cmapName)
        #
        fig = plt.figure(figsize=figsizeTup)
        gridC = round(100*legendWidth)
        gs = fig.add_gridspec(1,round(100*figsizeTup[0]-gridC),hspace=0,wspace=0)
        ax0 = fig.add_subplot(gs[:,:gridC])
        ax0.set_xticks([])
        ax0.set_yticks([])
        plt.setp(ax0.get_yticklabels(), visible=False)
        ax = fig.add_subplot(gs[:,gridC:],sharey=ax0)
        plt.setp(ax.get_yticklabels(), visible=False)
        #ax.get_yaxis().set_visible(False)
        for spine in ["left","right","top","bottom"]:
            ax.spines[spine].set_visible(False)
            ax0.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(left=leftW, right=rightW)
        ax.set_ylim(bottom=bottomW, top=topW)
        #
        # Graph section
        nodsizSr = (df_smi.loc[list(Gss.nodes)][nodeSizeCol] + sizeBase)*sizeScale
        nodcol = df_smi.loc[list(Gss.nodes)][nodeColorCol].values
        nn = nx.draw_networkx_nodes(Gss, pos, with_labels=True, ax=ax,
                            edgecolors=nodeEdgeColors, linewidths=nodeEdgeWeights,
                            node_size=nodsizSr.values,
                            node_color=nodcol, cmap=cm, alpha=1, vmin=vmin, vmax=vmax,
                            font_size=9, font_family='times new roman', font_weight='bold')
        if drawEdges:
            ee = nx.draw_networkx_edges(Gss, pos, ax=ax, width=(eWtArr-eWtArr.min())*3+0.6,
                            edge_color=eWtArr, edge_cmap=plt.cm.Greys,
                            edge_vmin=eWtArr.min()-0.5, edge_vmax=eWtArr.max()+0.1, alpha=edgeAlpha)
        # used with %matplotlib qt, for checking specific points
        if drawNodeLabels:
            df_smi['nodLab'] = df_smi.index
            if nodesShown is None:
                nx.draw_networkx_labels(Gss, pos, df_smi['nodLab'].reindex(Gss.nodes).to_dict(), font_size=font_size, font_family='Times New Roman')
            else:
                nx.draw_networkx_labels(Gss.subgraph(nodesShown), pos, df_smi['nodLab'].reindex(nodesShown).to_dict(), font_size=font_size, font_family='Times New Roman')
        # annotate specific nodes
        if len(annotateNodePos) > 0:
            for nod in annotateNodePos:
                coordx, coordy = pos[nod]
                ax.annotate(nod, xy=(coordx, coordy), xycoords='data', xytext=annotateNodePos[nod],
                textcoords=annotatePosStyle, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", linewidth=0.75),
                fontsize=font_size, family='Times New Roman')
        if markComm:
            theta = np.linspace(0, 2*np.pi, commEdgeInterpN, endpoint=False)
            circ = np.array([ratioAdjust*np.cos(theta),np.sin(theta)]).T
            self.nsg.calcComm()
            uList = self.nsg.filterComm(minCommSize)
            #clrs = ['orange','gold','yellow','lime','cyan',
            #'dodgerblue','blueviolet','magenta','crimson','olive']*10
            clrs = [commEdgeColor]*500
            Path = mpath.Path
            # Plot: convex hull
            for kc,clr in zip(uList, clrs[:len(uList)]):
                points0 = np.array([pos[k] for k in kc])
                try:
                    hull0 = ConvexHull(points0)
                    points1 = points0[hull0.vertices]
                    ptRadii = np.sqrt(nodsizSr.loc[np.array(list(kc))[hull0.vertices]]) + commEdgePad
                    points = np.repeat(points1, commEdgeInterpN, axis=0) + np.array([np.repeat(ptRadii, commEdgeInterpN).values]*2).T * np.concatenate([circ]*points1.shape[0],axis=0)
                except:
                    ptRadii = np.sqrt(nodsizSr.loc[np.array(list(kc))]) + commEdgePad
                    points = np.repeat(points0, commEdgeInterpN, axis=0) + np.array([np.repeat(ptRadii, commEdgeInterpN).values]*2).T * np.concatenate([circ]*2,axis=0)
                hull = ConvexHull(points)
                path_data = [(Path.MOVETO, tuple(points[hull.vertices[0]]))]
                # confines for cliques
                for simplex in hull.simplices:
                    ax.plot(points[simplex,0], points[simplex,1], clr, linestyle='dashed', linewidth=commEdgeWidth, alpha=1, zorder=2)
                # background colors filled (patch) for cliques
                for v in hull.vertices[1:]:
                    path_data.append((Path.LINETO, tuple(points[v])))
                path_data.append((Path.CLOSEPOLY, tuple(points[hull.vertices[0]])))
                codes, verts = zip(*path_data)
                path = mpath.Path(verts, codes)
                patch = mpatches.PathPatch(path, facecolor=clr, ec=clr, alpha=commAlpha, zorder=0, lw=0)
                ax.add_patch(patch)
        if len(groupsToMark) > 0:
            theta = np.linspace(0, 2*np.pi, commEdgeInterpN, endpoint=False)
            circ = np.array([ratioAdjust*np.cos(theta),np.sin(theta)]).T
            clrs = [commEdgeColor]*500
            Path = mpath.Path
            # Plot: convex hull
            for kc,clr in zip(groupsToMark, clrs[:len(groupsToMark)]):
                points0 = np.array([pos[k] for k in kc])
                try:
                    hull0 = ConvexHull(points0)
                    points1 = points0[hull0.vertices]
                    ptRadii = np.sqrt(nodsizSr.loc[np.array(list(kc))[hull0.vertices]]) + commEdgePad
                    points = np.repeat(points1, commEdgeInterpN, axis=0) + np.array([np.repeat(ptRadii, commEdgeInterpN).values]*2).T * np.concatenate([circ]*points1.shape[0],axis=0)
                except:
                    ptRadii = np.sqrt(nodsizSr.loc[np.array(list(kc))]) + commEdgePad
                    points = np.repeat(points0, commEdgeInterpN, axis=0) + np.array([np.repeat(ptRadii, commEdgeInterpN).values]*2).T * np.concatenate([circ]*2,axis=0)
                hull = ConvexHull(points)
                path_data = [(Path.MOVETO, tuple(points[hull.vertices[0]]))]
                # confines for cliques
                for simplex in hull.simplices:
                    ax.plot(points[simplex,0], points[simplex,1], clr, linestyle='dashed', linewidth=commEdgeWidth, alpha=1, zorder=2)
                # background colors filled (patch) for cliques
                for v in hull.vertices[1:]:
                    path_data.append((Path.LINETO, tuple(points[v])))
                path_data.append((Path.CLOSEPOLY, tuple(points[hull.vertices[0]])))
                codes, verts = zip(*path_data)
                path = mpath.Path(verts, codes)
                patch = mpatches.PathPatch(path, facecolor=clr, ec=clr, alpha=commAlpha, zorder=0, lw=0)
                ax.add_patch(patch)
        if showLegend:
            # --- color bar or legends
            if isContinuousValue:
                cax = inset_axes(ax0,
                width="50%",  # width = 10% of parent_bbox width
                height="35%",  # height : 50%
                loc='lower left',
                bbox_to_anchor=bboxTup,
                bbox_transform=ax0.transAxes, # (0.25,0.5,0.25,0.8) suggested
                borderpad=0,
                axes_kwargs={'frameon':False,'xticks':[9,6],'xticklabels':['1 nM',r'1 $\mu$M']})
                cbar = fig.colorbar(nn, cax, fraction=1)
                cbar.outline.set_visible(False)
            else:
                legend1 = ax0.legend(*nn.legend_elements(num=nLegend_activity), fontsize=9.5, loc="upper left",
                bbox_to_anchor=(0.05, 0.85),
                borderpad=0.3, handlelength=1.5, handletextpad=0.6, borderaxespad=0.3, labelspacing=0.6, frameon=False)#, title="Ranking")
                ax0.add_artist(legend1)
            # --- local discontinuity
            kw = dict(prop="sizes", num=nLegend_LD, color= 'darkgray', #nn.cmap(0.7),
            fmt="{x:.2f}", func=lambda s: s/sizeScale - sizeBase)
            #
            handles, labels = nn.legend_elements(**kw)
            legend2 = ax0.legend(handles, labels, fontsize=9.5, loc='upper left',
            bbox_to_anchor=(0.05, 0.40),
            borderpad=0.3, handlelength=1.5, handletextpad=0.6, borderaxespad=0.3, labelspacing=0.6, frameon=False)
        ax.set_aspect(ratioAdjust)
        fig.subplots_adjust(left=0.0, right=1.00, bottom=0.00, top=1.00)
        if savePng:
            PngName = PngNamePrefix+'NSG_{:s}_{:.2f}.png'.format(self.nsg.fpTypeStr,self.nsg.GsCutoff)
            fig.savefig(PngName, dpi=300)
    def pyvisHtml(self, sizeCol='localDiscontinuity', colorCol='pval', cmapName = 'RdYlGn_r', canvasHeight='600px', canvasWidth='800px'):
        df_train = self.nsg.df_train
        Gss = self.nsg.Gss
        #
        cm = plt.get_cmap(cmapName)
        yVal = df_train.loc[list(Gss.nodes), colorCol]
        cNorm = matplotlib.colors.Normalize(vmin=yVal.min(), vmax=yVal.max())
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        #
        sizeVal = df_train.loc[list(Gss.nodes), sizeCol]
        for v in Gss:
            Gss.nodes[v]['size'] = (sizeVal[v] + 0.05) * 30
            rgba = scalarMap.to_rgba(yVal[v])
            Gss.nodes[v]['color'] = 'rgb({:.2f},{:.2f},{:.2f})'.format(rgba[0]*255,rgba[1]*255,rgba[2]*255)
        self.ntHtml = Network(canvasHeight, canvasWidth)
        self.ntHtml.show_buttons(['physics'])
        self.ntHtml.from_nx(Gss)
        self.ntHtml.show('nx.html')
#




