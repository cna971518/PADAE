"""Measure controlled server load; synthetic uploads do not establish detection efficacy."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import numpy as np
from aggregators import aggregate_models, cma_weights
from configuration import CMA_PARAMETERS, LABELS, THRESHOLDS
from data_process import read_partition
from metrics import mqv_score
from mpdd import MPDD, first_layer_update
from utils import read_json, runtime_metadata, sha256_file, source_hashes, write_csv, write_json


class MemorySampler:
    def __init__(self, period=.01):
        import psutil
        self.process=psutil.Process(); self.period=period; self.samples=[]
        self.stop=threading.Event(); self.worker=None

    def __enter__(self):
        def sample():
            while not self.stop.is_set():
                self.samples.append(self.process.memory_info().rss)
                self.stop.wait(self.period)
        self.worker=threading.Thread(target=sample,daemon=True);self.worker.start()
        return self

    def __exit__(self,*args):
        self.stop.set();self.worker.join();self.samples.append(self.process.memory_info().rss)


def controlled_trial(dataset, data_root, models, candidates, repeats, trial, batch_size, output):
    if models<3 or not 1<=candidates<=models or repeats<1:
        raise ValueError('Require models>=3, 1<=candidates<=models and positive repeats.')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    from model import build_model
    from client import set_seed
    import tensorflow as tf
    path=Path(data_root)/dataset/'clients_10'/'Val.csv'
    data=read_partition(path,LABELS[dataset],batch_size=batch_size)
    set_seed(1000+trial)
    model=build_model(len(data.features),len(LABELS[dataset]),.01,'benchmark')
    global_weights=model.get_weights(); rng=np.random.default_rng(1000+trial)
    uploads=[[rng.normal(size=w.shape).astype(w.dtype) for w in global_weights] for _ in range(models)]
    tm,tr=THRESHOLDS[dataset];beta,lam=CMA_PARAMETERS[dataset];ids=list(range(candidates))
    model.set_weights(uploads[0]);model.predict(data.x,batch_size=batch_size,verbose=0)
    rows=[]
    for repeat in range(repeats):
        detector=MPDD(models,tr,monitor_only=True)
        deltas=[first_layer_update(uploads[i],global_weights) for i in ids]
        detector.inspect(ids,deltas,1);detector.inspect(ids,deltas,2)
        gpu_device='GPU:0' if tf.config.list_physical_devices('GPU') else None
        if gpu_device:
            try: tf.config.experimental.reset_memory_stats(gpu_device)
            except (ValueError, AttributeError): gpu_device=None
        with MemorySampler() as memory:
            start=time.perf_counter(); t0=start; scores={}
            for i,weights in enumerate(uploads):
                model.set_weights(weights)
                prediction=model.predict(data.x,batch_size=batch_size,verbose=0)
                scores[i]=mqv_score(data.y,prediction)
            mqv=time.perf_counter()-t0;t0=time.perf_counter()
            deltas=[first_layer_update(uploads[i],global_weights) for i in ids]
            extraction=time.perf_counter()-t0;t0=time.perf_counter()
            _,_,_,status,_=detector.inspect(ids,deltas,3)
            mpdd=time.perf_counter()-t0;t0=time.perf_counter()
            aggregate_models([uploads[i] for i in ids],cma_weights(ids,scores,beta,lam,'ceil'))
            cma=time.perf_counter()-t0;total=time.perf_counter()-start
        gpu_peak=None
        if gpu_device:
            try: gpu_peak=tf.config.experimental.get_memory_info(gpu_device)['peak']
            except (ValueError,AttributeError): pass
        rows.append({'dataset':dataset,'K':models,'controlled_candidates':candidates,'trial':trial,
                     'repeat':repeat,'validation_samples':data.used_rows,
                     'actual_mqv_passed':sum(v>=tm for v in scores.values()),
                     'gate_valid':status['gate_valid'],'gate_reason':status['gate_reason'],
                     'mqv_seconds':mqv,'extraction_seconds':extraction,'mpdd_seconds':mpdd,
                     'cma_seconds':cma,'total_seconds':total,'sampled_peak_rss_bytes':max(memory.samples),
                     'rss_sample_count':len(memory.samples),'tf_allocator_peak_bytes':gpu_peak,
                     'history_bytes':detector.sums.nbytes+detector.counts.nbytes,
                     'model_parameter_count':sum(w.size for w in global_weights),
                     'raw_model_payload_bytes':8*models*sum(w.size for w in global_weights)})
    write_json(output,{'protocol':'padae-controlled-load-v1','training_executed':False,
                       'synthetic_uploads':True,'controlled_pass_list':True,
                       'original_publication_benchmark_reproduced':False,
                       'environment':runtime_metadata(),'source_hashes':source_hashes(Path(__file__).parent),
                       'validation_sha256':sha256_file(path),'rows':rows})
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=tuple(LABELS),default=list(LABELS))
    p.add_argument('--models',nargs='+',type=int,default=[10,20,50,100,200,500,1000])
    p.add_argument('--candidate-ratio',type=float,default=1.)
    p.add_argument('--trials',type=int,default=3);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=500);p.add_argument('--data-root',default='data/processed')
    p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--trial-id',type=int,default=None,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if not 0<a.candidate_ratio<=1 or a.trials<1 or a.repeats<1 or any(n<3 for n in a.models):
        p.error('Invalid size, ratio or repetition count.')
    if a.trial_id is not None:
        if len(a.datasets)!=1 or len(a.models)!=1: p.error('One dataset and size per worker.')
        return controlled_trial(a.datasets[0],a.data_root,a.models[0],
                                max(1,int(a.models[0]*a.candidate_ratio)),a.repeats,a.trial_id,a.batch_size,a.output)
    tasks=[(d,k,t) for d in a.datasets for k in a.models for t in range(a.trials)]
    print(f'{len(tasks)} independent-process trials; execute={a.execute}')
    if not a.execute: return
    directory=Path(a.output)
    if directory.exists() and any(directory.iterdir()): raise FileExistsError(directory)
    directory.mkdir(parents=True,exist_ok=True)
    all_rows=[];trial_rows=[]
    for dataset,k,t in tasks:
        out=directory/f'{dataset}_K{k}_trial{t}.json'
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--datasets',dataset,'--models',str(k),
             '--data-root',a.data_root,'--repeats',str(a.repeats),'--batch-size',str(a.batch_size),
             '--candidate-ratio',str(a.candidate_ratio),'--output',str(out),'--trial-id',str(t)]
        subprocess.run(cmd,env=dict(os.environ,PYTHONHASHSEED=str(1000+t)),check=True)
        rows=read_json(out)['rows'];all_rows.extend(rows)
        trial_rows.append({'dataset':dataset,'K':k,'trial':t,
                           **{name:statistics.mean(row[name] for row in rows)
                              for name in ('mqv_seconds','mpdd_seconds','cma_seconds','extraction_seconds',
                                           'total_seconds','sampled_peak_rss_bytes')}})
    summary=[]
    for dataset in a.datasets:
        for k in a.models:
            group=[r for r in trial_rows if r['dataset']==dataset and r['K']==k]
            for metric in ('mqv_seconds','mpdd_seconds','cma_seconds','extraction_seconds',
                           'total_seconds','sampled_peak_rss_bytes'):
                values=[r[metric] for r in group]
                summary.append({'dataset':dataset,'K':k,'metric':metric,'n':len(values),
                                'mean':statistics.mean(values),
                                'sample_sd':statistics.stdev(values) if len(values)>1 else None})
    write_csv(directory/'timing_calls.csv',all_rows);write_csv(directory/'trial_means.csv',trial_rows)
    write_csv(directory/'summary.csv',summary)


if __name__=='__main__': main()
