import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score
from configuration import LABELS, RunConfig
from attacks import partial_data_tampering, label_flipping, random_weights, PDT_PAIRS
from metrics import mqv_score, classify
from aggregators import (aggregate_models,cma_weights,ddaba_weights,fedavg_weights,
                         coordinate_aggregate,trust_aggregate,middle_mean)
from mpdd import cosine_distance,first_layer_update,partition,MPDD
from preprocess.statavg import local_statistics,aggregate_statistics,normalize


@pytest.mark.parametrize('dataset',list(LABELS))
@pytest.mark.parametrize('ratio',[0,.5,1])
def test_pdt_immutable_donors(synthetic_xy,dataset,ratio):
    x,y=synthetic_xy(dataset);before=x.copy();labels=y.argmax(axis=1)
    poisoned,new_labels,events=partial_data_tampering(x,y,dataset,ratio,100)
    np.testing.assert_array_equal(x,before);np.testing.assert_array_equal(y,new_labels)
    changed=0
    for a,b,features in PDT_PAIRS[dataset]:
        ai,bi=LABELS[dataset].index(a),LABELS[dataset].index(b)
        for source,other in [(ai,bi),(bi,ai)]:
            rows=np.flatnonzero(labels==source);mask=np.any(poisoned[np.ix_(rows,features)]!=x[np.ix_(rows,features)],axis=1)
            assert mask.sum()==int(len(rows)*ratio)
            for row in rows[mask]:
                np.testing.assert_array_equal(poisoned[row,features],x[labels==other][0,features])
                outside=[j for j in range(x.shape[1]) if j not in features]
                np.testing.assert_array_equal(poisoned[row,outside],x[row,outside])
            changed+=int(mask.sum())
    assert changed==len(x)*ratio


@pytest.mark.parametrize('dataset',list(LABELS))
@pytest.mark.parametrize('ratio',[0,.5,1])
def test_label_flip_preserves_features(synthetic_xy,dataset,ratio):
    x,y=synthetic_xy(dataset);out,labels,events=label_flipping(x,y,dataset,ratio,12)
    np.testing.assert_array_equal(x,out)
    assert np.sum(labels.argmax(1)!=y.argmax(1))==int(len(x)*ratio)
    assert np.all(labels.sum(1)==1)
    np.testing.assert_array_equal(label_flipping(x,y,dataset,ratio,12)[1],labels)


def test_rw_replaces_biases_and_is_seeded():
    weights=[np.full((100,5),50,dtype=np.float32),np.full(8,50,dtype=np.float32)]
    output=random_weights(weights,100)
    assert len(output)==2 and all(a.dtype==np.float32 for a in output)
    assert all(np.all(a!=50) for a in output)
    for a,b in zip(output,random_weights(weights,100)):np.testing.assert_array_equal(a,b)
    assert .8<output[0].std()<1.2


@pytest.mark.parametrize('seed',range(5))
def test_mqv_equals_empirical_ks(seed):
    rng=np.random.default_rng(seed);c=6
    y=np.eye(c)[rng.integers(c,size=100)];pred=rng.dirichlet(np.ones(c),size=100)
    expected=ks_2samp(pred[y.astype(bool)],pred[~y.astype(bool)]).statistic
    assert mqv_score(y,pred)==pytest.approx(expected)


def test_mqv_uniform_and_perfect():
    y=np.eye(4)[[0,1,2,3]]
    assert mqv_score(y,np.full_like(y,.25))==0
    assert mqv_score(y,y)==1


def test_class_metrics_all_classes():
    y=np.eye(4)[[0,1,2,3,1]];p=np.eye(4)[[0,2,2,3,0]]
    scores,rows,cm=classify(y,p,('a','b','c','d'))
    assert scores['accuracy']==60
    assert scores['macro_f1']==pytest.approx(100*f1_score(y.argmax(1),p.argmax(1),average='macro'))
    assert cm.sum()==5 and sum(r['support'] for r in rows)==5


@pytest.mark.parametrize('a,b,expected',[( [0,0],[0,0],0),([0,0],[1,0],1),([1,0],[1,0],0),([1,0],[-1,0],2),([1,0],[0,1],1)])
def test_cosine_zero_and_direction(a,b,expected): assert cosine_distance(a,b)==expected


def test_first_kernel_not_bias():
    a=[np.ones((3,4)),np.full(4,100)]
    g=[np.full((3,4),.5),np.zeros(4)]
    np.testing.assert_array_equal(first_layer_update(a,g),np.full(12,.5))


def test_partition_formula():
    d=np.full((5,5),2.);np.fill_diagonal(d,0)
    d[:4,:4]=.2;np.fill_diagonal(d,0)
    result=partition(d,range(5))
    assert result.valid and result.suspicious==(4,)
    assert result.R==pytest.approx(10)
    assert result.q[4]==pytest.approx(10)


def test_equal_groups_invalid():
    d=np.array([[0,.1,2,2],[.1,0,2,2],[2,2,0,.1],[2,2,.1,0]])
    assert partition(d,range(4)).reason=='equal_size_clusters'


def vectors(n=10):return [np.array([1.,.01*i]) for i in range(n-1)]+[np.array([-1.,0])]


def test_mpdd_warmup_warning_confirmation_and_reset():
    model=MPDD(10,1.1,record_pairs=True)
    ids=list(range(10));v=vectors()
    for r in (1,2):
        kept,removed,warning,row,pairs=model.inspect(ids,v,r)
        assert row['warmup'] and not removed and not warning
    kept,removed,warning,row,pairs=model.inspect(ids,v,3)
    assert warning==[9] and not removed and 9 in kept
    kept,removed,warning,row,pairs=model.inspect(ids,v,4)
    assert removed==[9] and 9 not in kept and row['episode_round']==4
    kept,removed,warning,row,pairs=model.inspect(ids[:-1],v[:-1],5)
    assert row['warmup'] and row['episode_round']==1 and row['episode']==2
    assert all(p['observations']==1 for p in pairs)


def test_membership_change_clears_hits():
    model=MPDD(10,1.1);ids=list(range(10));v=vectors()
    for r in range(1,4): model.inspect(ids,v,r)
    assert model.hits[9]==1
    *_,row,pairs=model.inspect(ids[1:],v[1:],4)
    assert row['warmup'] and not model.hits.any()


def test_minimum_survivors_applies_to_candidates():
    model=MPDD(10,1.1);ids=list(range(6));v=vectors(6)
    for r in range(1,5):retained,removed,warning,row,_=model.inspect(ids,v,r)
    assert not removed and row['safety_blocked'] and len(retained)==6


def test_monitor_never_removes():
    model=MPDD(10,1.1,monitor_only=True)
    for r in range(1,7): kept,removed,warning,row,_=model.inspect(list(range(10)),vectors(),r)
    assert len(kept)==10 and not removed and row['proposed_ids']==[9]


def test_cma_group_mass_and_rounding():
    ids=list(range(6));scores={i:float(i) for i in ids}
    ceil=cma_weights(ids,scores,.2,.9,'ceil')
    assert ceil[-2:].sum()==pytest.approx(.9) and ceil[:-2].sum()==pytest.approx(.1)
    with pytest.raises(ValueError): cma_weights(ids,scores,.2,.9,'floor')
    np.testing.assert_allclose(cma_weights([0],{0:1},.2,.5),[1])
    ties=cma_weights(ids,{i:1 for i in ids},.2,.8)
    assert ties[0]==pytest.approx(.4) and ties[1]==pytest.approx(.4)


def test_aggregate_includes_bias_and_fedavg_counts():
    models=[[np.ones((2,2),dtype=np.float32),np.ones(2,dtype=np.float32)*2],
            [np.ones((2,2),dtype=np.float32)*3,np.ones(2,dtype=np.float32)*4]]
    w=fedavg_weights([1,3]);out=aggregate_models(models,w)
    np.testing.assert_allclose(out[0],2.5);np.testing.assert_allclose(out[1],3.5)


@pytest.mark.parametrize('method',['median','trimmed_mean'])
def test_coordinate_outlier(method):
    models=[[np.array([a],dtype=np.float32)] for a in (1,2,3,4,100)]
    assert coordinate_aggregate(models,method)[0][0]==3


def test_ddaba_equal_and_outlier():
    w,_=ddaba_weights([.9]*5);np.testing.assert_allclose(w,.2)
    w,info=ddaba_weights([.99,.98,.97,.96,.2])
    assert w.sum()==pytest.approx(1) and w[-1]==0 and info['rule']=='text_consistent_plus'
    assert w[0]==pytest.approx(w[1])
    assert w[0]==pytest.approx(2*w[2])


@pytest.mark.parametrize('method',['fltrust','flest'])
def test_trust_zero_root_keeps_global(method):
    g=[np.zeros(2,dtype=np.float32)];models=[[np.ones(2,dtype=np.float32)],[np.ones(2,dtype=np.float32)*2]]
    out,w,info=trust_aggregate(models,g,g,method,100)
    np.testing.assert_array_equal(out[0],g[0]);assert w.sum()==0


def test_trust_antiparallel_has_no_weight():
    g=[np.zeros(2,dtype=np.float32)];root=[np.array([1,0],dtype=np.float32)]
    models=[[np.array([2,0],dtype=np.float32)],[np.array([-1,0],dtype=np.float32)]]
    out,w,info=trust_aggregate(models,g,root,'fltrust',100)
    np.testing.assert_allclose(w,[1,0]);np.testing.assert_allclose(out[0],[1,0])


def test_statavg_matches_pooled_distribution():
    a=np.array([[1,2,5],[2,4,5]],float);b=np.array([[10,8,5],[20,10,5],[30,12,5]],float)
    stats=aggregate_statistics([local_statistics(a),local_statistics(b)])
    pool=np.concatenate([a,b]);np.testing.assert_allclose(stats['mean'],pool.mean(0))
    np.testing.assert_allclose(stats['variance'],pool.var(0))
    transformed=normalize(pool,stats);np.testing.assert_allclose(transformed.mean(0),0,atol=1e-7)
    np.testing.assert_array_equal(transformed[:,2],0)
    np.testing.assert_array_equal(normalize(np.array([[999,999,999.]]),stats)[:,2],0)


def test_configuration_fixed_parameters():
    for dataset,expected in [('UNSW-NB15',(.4,.2,.5)),('CIC-IDS2017',(.7,.2,.9))]:
        cfg=RunConfig(dataset=dataset).resolved()
        assert (cfg.tau_mqv,cfg.beta,cfg.lam)==expected
        assert len(cfg.malicious_ids)==0
    cfg=RunConfig(scenario='pdt_0p5').resolved();assert cfg.malicious_ids==(0,1,2,3)


@pytest.mark.parametrize('field,value',[('tau_mqv',1.1),('tau_r',1.),('beta',0),('lam',1.1),('rounds',0),('batch_size',0)])
def test_bad_configuration_rejected(field,value):
    with pytest.raises(ValueError): RunConfig(**{field:value}).resolved()
