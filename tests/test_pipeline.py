from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from conftest import FakeModel
from configuration import RunConfig, METHODS, LABELS
from data_process import load_data,read_partition
from preprocess.prepare import fit_transformer,transform,prepare_selected
from server import FederatedRunner
from run_experiments import make_plan,verified_done
from summarize_results import summarize
from calibrate_mpdd import calibration_candidates
from utils import read_json,write_json


def test_full_independent_test_and_fixed_splits(fixed_data):
    root=fixed_data();cfg=RunConfig(data_root=str(root),batch_size=10).resolved()
    data=load_data(cfg)
    assert data.validation.used_rows==70
    assert data.test.used_rows==73 and data.test.raw_rows==73
    assert data.train[0].used_rows==70
    assert load_data(replace(cfg,seed=104)).manifest['fingerprint']==data.manifest['fingerprint']


def test_data_hash_verification(fixed_data,tmp_path):
    root=fixed_data();cfg=RunConfig(data_root=str(root),batch_size=10).resolved();bundle=load_data(cfg)
    path=tmp_path/'manifest.json';write_json(path,bundle.manifest)
    load_data(replace(cfg,expected_data_manifest=str(path)))
    datafile=cfg.client_dir/'client_01_train.csv'
    frame=pd.read_csv(datafile);frame.iloc[0,0]+=1;frame.to_csv(datafile,index=False)
    with pytest.raises(ValueError,match='hashes'):load_data(replace(cfg,expected_data_manifest=str(path)))


def test_schema_mismatch_fails(fixed_data):
    root=fixed_data();cfg=RunConfig(data_root=str(root),batch_size=10).resolved()
    path=cfg.client_dir/'Val.csv';frame=pd.read_csv(path);columns=list(frame);columns[0],columns[1]=columns[1],columns[0]
    frame[columns].to_csv(path,index=False)
    with pytest.raises(ValueError,match='order'):load_data(cfg)


def test_no_legacy_resplit_fallback(tmp_path):
    directory=tmp_path/'UNSW-NB15'/'clients_10';directory.mkdir(parents=True)
    (directory/'client_01.csv').write_text('old\n1\n')
    with pytest.raises(FileNotFoundError,match='client_01_train'):load_data(RunConfig(data_root=str(tmp_path)).resolved())


def test_encoder_not_fitted_on_heldout():
    train=pd.DataFrame({'f':[1.,2.,3.,4.], 'proto':['tcp','udp']*2,
                        'state':['s','t']*2,'service':['http','ftp']*2})
    transformer=fit_transformer(train,'UNSW-NB15')
    held=train.iloc[:1].copy();held['proto']='unseen'
    transform(held,transformer)
    assert 'unseen' not in transformer['categorical']['proto']['encoder'].categories_[0]


@pytest.mark.parametrize('dataset',list(LABELS))
def test_synthetic_preprocessing_partition_disjoint(tmp_path,dataset):
    classes=LABELS[dataset];rng=np.random.default_rng(9);count=300*len(classes)
    target='attack_cat' if dataset=='UNSW-NB15' else 'Label'
    frame=pd.DataFrame(rng.normal(size=(count,64)),columns=[f'f{i}' for i in range(64)])
    frame[target]=np.repeat(classes,300)
    if dataset=='UNSW-NB15':
        frame['label']=(frame[target]!='normal').astype(int)
        frame['proto']=np.resize(['tcp','udp','icmp'],count)
        frame['state']=np.resize(['fin','con'],count)
        frame['service']=np.resize(['dns','http'],count)
    prepare_selected(frame,dataset,[10],tmp_path)
    directory=tmp_path/dataset/'clients_10'
    members=pd.read_csv(directory/'split_membership.csv')
    assert len(members)==len(frame) and not members.row_id.duplicated().any()
    manifest=read_json(directory/'preprocessing_manifest.json')
    assert manifest['held_out_used_for_fitting'] is False
    data=load_data(RunConfig(dataset=dataset,data_root=str(tmp_path),batch_size=2).resolved())
    assert data.test.used_rows==len(pd.read_csv(directory/'Test.csv'))


@pytest.mark.parametrize('method',METHODS)
def test_runner_control_flow_all_methods(tmp_path,bundle,method):
    FakeModel.rejected_ids=set()
    cfg=RunConfig(method=method,rounds=4,local_epochs=1,batch_size=10,save_model=False,
                  output=str(tmp_path/method)).resolved()
    runner=FederatedRunner(cfg,bundle,model_factory=FakeModel,seed_function=lambda s:None)
    result=runner.run()
    assert result['status']=='completed' and result['completed_rounds']==4
    assert result['samples']==bundle.test.used_rows
    assert all(m.fits==4 for m in runner.local_models)
    assert verified_done(cfg.output,cfg.as_dict())
    assert read_json(Path(cfg.output)/'result.json')['accuracy']==100


def test_all_reject_boundary_not_success(tmp_path,bundle):
    FakeModel.rejected_ids=set(range(10))
    cfg=RunConfig(rounds=4,local_epochs=1,batch_size=10,save_model=False,output=str(tmp_path/'reject')).resolved()
    runner=FederatedRunner(cfg,bundle,FakeModel,lambda s:None);result=runner.run()
    assert result['status']=='all_mqv_rejected' and result['completed_rounds']==0
    assert result['mqv_ever_frr']==100 and result['removed_benign']==0
    FakeModel.rejected_ids=set()


def test_mqv_two_rejections_remove_without_mixed_streak(tmp_path,bundle):
    FakeModel.rejected_ids={0}
    cfg=RunConfig(method='mqv_only',scenario='lf_0p5',rounds=4,local_epochs=1,batch_size=10,
                  save_model=False,output=str(tmp_path/'mqv')).resolved()
    runner=FederatedRunner(cfg,bundle,FakeModel,lambda s:None);result=runner.run()
    assert runner.local_models[0].fits==2 and runner.local_models[1].fits==4
    assert result['mqv_removed_malicious']==1 and result['mpdd_removed_malicious']==0
    status=pd.read_csv(Path(cfg.output)/'client_final_status.csv')
    assert status.iloc[0].removed_round==2
    FakeModel.rejected_ids=set()


def test_mqv_equality_passes(tmp_path,bundle):
    FakeModel.rejected_ids=set(range(10))
    cfg=RunConfig(method='mqv_only',tau_mqv=0,rounds=2,local_epochs=1,batch_size=10,
                  save_model=False,output=str(tmp_path/'equal')).resolved()
    result=FederatedRunner(cfg,bundle,FakeModel,lambda s:None).run()
    assert result['completed_rounds']==2 and result['mqv_ever_frr']==0
    FakeModel.rejected_ids=set()


def test_output_not_overwritten(tmp_path,bundle):
    out=tmp_path/'exists';out.mkdir();(out/'keep.txt').write_text('original')
    runner=FederatedRunner(RunConfig(output=str(out)),bundle,FakeModel,lambda s:None)
    with pytest.raises(FileExistsError):runner.run()
    assert not runner.output_owned and (out/'keep.txt').read_text()=='original'


@pytest.mark.parametrize('preset,count',[('comparison',1260),('ablation',400),('clean-ablation',80),
                                        ('mqv-sensitivity',210),('cma-grid',250),('calibration',20)])
def test_plan_counts(preset,count): assert len(make_plan(preset))==count


def test_one_dataset_ablation_200():assert len(make_plan('ablation',datasets=['UNSW-NB15']))==200


def test_summary_recomputes_sample_sd(tmp_path,bundle):
    FakeModel.rejected_ids=set()
    root=tmp_path/'runs'
    for seed in (100,101):
        cfg=RunConfig(method='uniform',seed=seed,rounds=1,local_epochs=1,batch_size=10,
                      save_model=False,output=str(root/f's{seed}')).resolved()
        FederatedRunner(cfg,bundle,FakeModel,lambda s:None).run()
    report=summarize([root],[100,101],tmp_path/'summary')
    assert report['complete_groups']==1 and not report['incomplete_groups']
    frame=pd.read_csv(tmp_path/'summary'/'global_summary.csv')
    assert frame.loc[frame.metric=='accuracy','mean'].iloc[0]==100
    assert frame.loc[frame.metric=='accuracy','sample_sd'].iloc[0]==0
    report=summarize([root],[100,101,102],tmp_path/'incomplete')
    assert report['complete_groups']==0 and len(report['incomplete_groups'])==1


def test_calibration_uses_same_episode_consecutive_rounds():
    rows=[{'round':1,'episode':1,'warmup':True,'structural_valid':True,'R':8.,'suspicious_ids':[0]},
          {'round':2,'episode':1,'warmup':True,'structural_valid':True,'R':7.,'suspicious_ids':[0]},
          {'round':3,'episode':1,'warmup':False,'structural_valid':True,'R':1.2,'suspicious_ids':[0]},
          {'round':4,'episode':1,'warmup':False,'structural_valid':True,'R':1.3,'suspicious_ids':[0]},
          {'round':5,'episode':2,'warmup':False,'structural_valid':True,'R':9.,'suspicious_ids':[0]}]
    values=calibration_candidates(rows)
    assert len(values)==1 and values[0]['critical_R']==1.2 and values[0]['rounds']==[3,4]


def test_cli_check_only_without_tensorflow(tmp_path,fixed_data):
    root=fixed_data();report=tmp_path/'check.json'
    code=subprocess.run([sys.executable,'-B',str(Path(__file__).parents[1]/'main.py'),
                         '--data-root',str(root),'--batch-size','10','--check-only','--check-report',str(report)],
                        capture_output=True,text=True)
    assert code.returncode==0,code.stderr
    assert read_json(report)['training_executed'] is False


def test_plan_identity_override_rejected():
    with pytest.raises(ValueError,match='Identity'):
        make_plan('comparison',overrides={'scenario':'clean'})


def test_summary_detects_changed_output(tmp_path,bundle):
    FakeModel.rejected_ids=set()
    cfg=RunConfig(method='uniform',rounds=1,local_epochs=1,batch_size=10,save_model=False,
                  output=str(tmp_path/'run')).resolved()
    FederatedRunner(cfg,bundle,FakeModel,lambda s:None).run()
    path=Path(cfg.output)/'result.json';data=read_json(path);data['accuracy']=0;write_json(path,data)
    with pytest.raises(ValueError,match='integrity'):verified_done(cfg.output,cfg.as_dict())


def test_old_pvalue_flag_not_silently_accepted():
    from args import parse_args
    with pytest.raises(SystemExit):parse_args(['--pvalue_threshold','0.05'])


def test_invalid_boolean_config_rejected():
    with pytest.raises(ValueError,match='boolean'):RunConfig(trim_batches='false').resolved()


@pytest.mark.parametrize('setting,K,bad',[('1-out-of-10',10,1),('4-out-of-10',10,4),('4-out-of-20',20,4)])
def test_three_client_settings(fixed_data,setting,K,bad):
    root=fixed_data(K=K)
    cfg=RunConfig(data_root=str(root),client_setting=setting,scenario='pdt_0p5',batch_size=10).resolved()
    data=load_data(cfg)
    assert len(data.train)==K and len(cfg.malicious_ids)==bad and len(data.manifest['files'])==2*K+2
    assert data.test.used_rows==73


def test_untrimmed_mode_is_explicit(fixed_data):
    root=fixed_data()
    cfg=RunConfig(data_root=str(root),batch_size=10,trim_batches=False).resolved()
    data=load_data(cfg)
    assert data.train[0].used_rows==72 and data.validation.used_rows==72
