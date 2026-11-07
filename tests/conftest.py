"""Synthetic fixtures; none of these tests reproduces paper accuracy."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from configuration import LABELS
from data_process import DataBundle, PartitionData


@pytest.fixture
def synthetic_xy():
    def build(dataset='UNSW-NB15',per_class=20):
        count=len(LABELS[dataset]);y=np.eye(count,dtype=np.float32)[np.repeat(np.arange(count),per_class)]
        x=np.repeat(np.arange(count),per_class)[:,None]*100+np.arange(64)[None,:]
        return x.astype(np.float32),y
    return build


@pytest.fixture
def fixed_data(tmp_path,synthetic_xy):
    def build(dataset='UNSW-NB15',K=10):
        directory=tmp_path/'data'/dataset/f'clients_{K}';directory.mkdir(parents=True)
        x,y=synthetic_xy(dataset,12)
        features=[f'f{i}' for i in range(x.shape[1])]
        for name in [f'client_{i:02d}_{part}.csv' for i in range(1,K+1) for part in ('train','test')]+['Val.csv','Test.csv']:
            frame=pd.DataFrame(x,columns=features);frame[list(LABELS[dataset])]=y
            if name=='Test.csv': frame=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
            frame.to_csv(directory/name,index=False)
        return tmp_path/'data'
    return build


@pytest.fixture
def bundle(synthetic_xy):
    x,y=synthetic_xy();x=x/100;x[:,0]=y.argmax(axis=1)
    part=PartitionData(x,y,tuple(f'f{i}' for i in range(x.shape[1])),len(x),len(x))
    return DataBundle([part]*10,[part]*10,part,part,{'fingerprint':'synthetic-test-only'})


class FakeModel:
    """Control-flow test double, not a learned classifier."""
    rejected_ids=set()
    def __init__(self,input_dim,output_dim,learning_rate,name):
        self.name=name;self.classes=output_dim;self.fits=0
        self.weights=[np.zeros((input_dim,2),dtype=np.float32),np.zeros(output_dim,dtype=np.float32)]
    def get_weights(self):return [a.copy() for a in self.weights]
    def set_weights(self,values):self.weights=[a.copy() for a in values]
    def fit(self,x,y,**kwargs):
        assert kwargs['shuffle'] is False
        self.fits+=1
        self.weights[0]+=np.float32(.01)
    def predict(self,x,**kwargs):
        client=int(self.name.split('_')[-1])-1 if self.name.startswith('client_') else None
        if client in self.rejected_ids:
            return np.full((len(x),self.classes),1/self.classes,dtype=np.float32)
        labels=np.asarray(x[:,0],dtype=int)%self.classes
        p=np.full((len(x),self.classes),.05/(self.classes-1),dtype=np.float32)
        p[np.arange(len(x)),labels]=.95
        return p
