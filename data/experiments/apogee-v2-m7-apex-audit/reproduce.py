"""Offline apogee audit. Run from the PSIntelligence root.

Modes: apex | states | current | sim_r_2p25.
Except for apex, pass the path to a second ukf_replay executable built with
ukf_sim_r.patch applied to a COPY of ukf.cpp. The normal replay executable
and existing model bundles are read from the paths below. No target changes.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path.cwd()))
from scripts import benchmark_apogee as b
from apogee_models import GRUApogeeModel, score_state_dict

ROOT = Path('data/experiments')
OUT = ROOT / 'apogee-v2-m7-apex-audit'
OUT.mkdir(exist_ok=True)
RUN = 'w100_gru_h32_l1_seed42_raw'
OLD = ROOT / 'apogee-v2-m7-ukf' / RUN
COAST = ROOT / 'apogee-v2-m7-ukf-coasting' / RUN
CFG = json.loads((COAST / 'run.json').read_text())
F = b._read_data(Path(CFG['data'])).sort_values(['flight_id', 'time']).reset_index(drop=True)
M = pd.read_csv('data/simulated/flight_metadata.csv.gz').set_index('flight_id')
PAD = F.groupby('flight_id')['altitude'].first() - F.groupby('flight_id')['altitude_rel'].first()
M['true_apogee'] = M.apogee_altitude - PAD
assert (M.true_apogee - F.groupby('flight_id').apogee_rel.first()).abs().max() < 0.001
CLEAN = b.ap.resample_causal(F, columns=('altitude_rel', 'velocity_z')).rename(
    columns={'time': 'endpoint_time', 'altitude_rel': 'true_h', 'velocity_z': 'true_v'})
NORMAL = Path('../../Avionics/flight-computer/.pio/tools/ukf_replay.exe')
assert b.ap.sha256_file(NORMAL) == CFG['ukf_replay_sha256']
assert b.ap.sha256_file(Path(CFG['data'])) == CFG['data_sha256']


def data(replay):
    return b._benchmark_inputs(F, SimpleNamespace(state_source='m7_ukf', ukf_replay=replay))


def coast_before_apex(ep):
    ep = b._select_endpoints(F, ep.loc[np.isfinite(ep.target)], 'coasting')
    apex = ep.flight_id.map(M.apogee_time)
    return ep.loc[ep.endpoint_time <= apex]


def sample(feat, ep, split):
    ids = CFG['split'][split]
    x, h, v, _, fi = b._samples_for_window(feat, ep, ids, 100)
    rows = pd.concat([ep[(ep.flight_id == i) & ep.window_valid] for i in ids], ignore_index=True)
    np.testing.assert_array_equal(rows.flight_id, fi)
    rows = rows.merge(CLEAN[['flight_id','endpoint_time','true_h','true_v']],
                      on=['flight_id','endpoint_time'], validate='one_to_one').join(
        M[['apogee_time','burnout_time','true_apogee']], on='flight_id', validate='many_to_one')
    assert np.array_equal(rows.flight_id, fi) and len(rows) == len(x)
    rows['tts'] = rows.apogee_time - rows.endpoint_time
    rows['since_burnout'] = rows.endpoint_time - rows.burnout_time
    assert rows.tts.ge(0).all()
    return x, h, v, rows.true_apogee.to_numpy(dtype=np.float32), rows


def subsets(rows):
    return {'coast': rows.tts.notna(), 'coast_first_1s': rows.since_burnout.lt(1),
            'last_1s': rows.tts.le(1), 'last_500ms': rows.tts.le(.5),
            'last_200ms': rows.tts.le(.2), 'last_per_flight': ~rows.flight_id.duplicated(keep='last')}


def summary(rows, value):
    a = np.asarray(value, dtype=float)
    assert len(a) == len(rows) and np.isfinite(a).all()
    for label, mask in subsets(rows).items():
        e = a[mask]
        if len(e):
            yield {'region': label, 'n': len(e), 'mae': np.abs(e).mean(),
                   'p95': np.quantile(np.abs(e), .95), 'max': np.abs(e).max(),
                   'bias': e.mean(), 'over_5m': (np.abs(e)>5).sum()}


def native(path, x):
    scaler = joblib.load(path / 'scaler.pkl')
    model = GRUApogeeModel(hidden_dim=32, num_layers=1)
    model.load_state_dict(torch.load(path / 'model.pth', weights_only=True, map_location='cpu'))
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(b._scale_array(scaler, x)))[0].numpy()


def assess_apex(feat, ep):
    stats = []
    for stride in [10, 1]:
        e = ep if stride == 10 else b.ap.make_endpoint_manifest(feat, stride=1)
        e = coast_before_apex(e)
        x,h,v,y,rows = sample(feat,e,'test')
        rows = rows[['flight_id','endpoint_time','true_apogee','tts','since_burnout']].copy()
        den = h + v*v/(2*b.ap.SIM_G)
        for model, path in [('all_phase_r', OLD), ('coasting_r', COAST)]:
            rows[model] = native(path, x)*den
            stats += [dict(model=model, cadence_ms=stride*20, **s) for s in summary(rows, rows[model]-y)]
        last = rows.loc[~rows.flight_id.duplicated(keep='last'), 'tts']
        print('cadence',stride*20,'last lead min/max',last.min(),last.max(),flush=True)
        rows.to_csv(OUT / f'existing_predictions_{stride*20}ms.csv',index=False)
    tab = pd.DataFrame(stats)
    tab.to_csv(OUT/'existing_summary.csv',index=False)
    print(tab.to_string(index=False),flush=True)


def compare_states(datasets, eps):
    stats = []
    for name,(feat,_),ep in zip(['current','sim_r_2p25'],datasets,eps):
        x,h,v,y,rows = sample(feat,ep,'test')
        hc,vc = rows.true_h.to_numpy(), rows.true_v.to_numpy()
        h_err = h-hc
        v_err = v-vc
        gain_err = (v*v-vc*vc)/(2*b.ap.SIM_G)
        for term,value in [('h_error_m',h_err),('vz_error_mps',v_err),('velocity_term_error_m',gain_err),
                           ('denominator_error_m',h_err+gain_err)]:
            stats += [dict(ukf=name, term=term, **s) for s in summary(rows,value)]
        if name == 'current':
            r = native(COAST, x)
            d = h+v*v/(2*b.ap.SIM_G)
            for term,value in [('frozen_head_delta_h_m',r*h_err), ('frozen_head_delta_v_m',r*gain_err),
                               ('native_error_m',d*r-y),('frozen_r_clean_head_error_m',r*(hc+vc*vc/(2*b.ap.SIM_G))-y)]:
                stats += [dict(ukf=name, term=term, **s) for s in summary(rows,value)]
    tab=pd.DataFrame(stats)
    tab.to_csv(OUT/'ukf_state_summary.csv',index=False)
    print(tab[tab.region.isin(['coast','coast_first_1s','last_1s','last_per_flight'])].to_string(index=False),flush=True)


class Comparator(nn.Module):
    def __init__(self, kind, center):
        super().__init__()
        self.gru = nn.GRU(6,32,batch_first=True)
        self.fc = nn.Sequential(nn.Linear(34,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
        self.kind = kind
        self.center = center

    def forward(self,x,h,v):
        # Both heads get identical six-channel histories and endpoint h/v.
        _, hidden = self.gru(x)
        z = self.fc(torch.cat((hidden[0], (h/100).unsqueeze(-1), (v/50).unsqueeze(-1)),dim=1)).squeeze(-1)
        if self.kind == 'direct':
            return self.center + 100*z
        return (h+v*v/(2*b.ap.SIM_G))*(1+0.5*torch.tanh(z))


def train(kind,seed,train_data,val_data):
    torch.manual_seed(seed)
    center=float(train_data[3].mean())
    model=Comparator(kind,center)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lr=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5)
    best, payload, best_epoch = float('inf'), None, 0
    x,h,v,y=train_data
    xv,hv,vv,yv=val_data
    for epoch in range(1,121):
        model.train()
        for idx in torch.randperm(len(x)).split(128):
            opt.zero_grad()
            err=(model(x[idx],h[idx],v[idx])-y[idx])/100
            err.square().mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.eval()
        with torch.inference_mode():
            loss=float(((model(xv,hv,vv)-yv)/100).square().mean())
        lr.step(loss)
        if loss<best:
            best,payload,best_epoch=loss,score_state_dict(model.state_dict()),epoch
        elif epoch-best_epoch>=12:
            break
    assert payload is not None
    model.load_state_dict(payload)
    return model.eval(),best_epoch


def ablate(feat, common_ep, state, seeds):
    # Shares the time grid and set of viable windows between both UKFs.
    sc=b._fit_endpoint_scaler(feat,common_ep,CFG['split']['train'])
    sets, indices = {},{}
    for split in ('train','validation','test'):
        x,h,v,y,rows=sample(feat,common_ep,split)
        sets[split]=tuple(torch.from_numpy(a) for a in (b._scale_array(sc,x),h,v,y))
        indices[split]=rows
        print(state,split,len(x),flush=True)
    out, prediction_rows = [], []
    for seed in seeds:
        for kind in ('r_height_loss','direct'):
            model,epoch=train(kind,seed,sets['train'],sets['validation'])
            print('trained',state,kind,seed,'best epoch',epoch,flush=True)
            for split in ('validation','test'):
                x,h,v,y=sets[split]
                with torch.inference_mode():
                    prediction=model(x,h,v).numpy()
                rows=indices[split]; err=prediction-rows.true_apogee.to_numpy()
                out.extend(dict(ukf=state, model=kind, seed=seed, best_epoch=epoch, split=split, **s) for s in summary(rows,err))
                prediction_rows.append(rows[['flight_id','endpoint_time','tts','since_burnout','true_apogee']].assign(
                    ukf=state,model=kind,seed=seed,split=split,prediction=prediction))
    tab=pd.DataFrame(out)
    tab.to_csv(OUT/f'head_summary_{state}.csv',index=False)
    pd.concat(prediction_rows,ignore_index=True).to_csv(OUT/f'head_predictions_{state}.csv',index=False)
    print(tab[tab.region.isin(['coast','coast_first_1s','last_1s','last_per_flight'])].to_string(index=False),flush=True)


def main():
    mode=sys.argv[1]
    cur=data(NORMAL)
    if mode=='apex':
        assess_apex(*cur)
        return
    modified=Path(sys.argv[2]); tuned=data(modified)
    candidates=[coast_before_apex(e).loc[lambda ep: ep.window_valid] for _,e in (cur,tuned)]
    k=['flight_id','endpoint_index','endpoint_time']
    common=candidates[0][k].merge(candidates[1][k],on=k,validate='one_to_one')
    eps=[e.merge(common,on=k,validate='one_to_one') for e in candidates]
    if mode=='states':
        compare_states((cur,tuned),eps)
        (OUT/'audit_config.json').write_text(json.dumps({
            'data':CFG['data'],'data_sha256':CFG['data_sha256'],'split':CFG['split'],'window':100,
            'replay_sha256':{'current':CFG['ukf_replay_sha256'],'sim_r_2p25':b.ap.sha256_file(modified)},
            'baro_R_m2':{'current':0.0217839917878801,'sim_r_2p25':2.25},'R_experiment':'host only; Q and model unchanged',
            'head_kind':['r_height_loss','direct'],'head_training':'apogee MSE in units of 100 m for both; same GRU and endpoint h/v inputs',
            'seeds':[42,43],'max_epochs':120,'stride':10,'phase':'coasting endpoint only'},indent=2))
    else:
        chosen={'current':0,'sim_r_2p25':1}[mode]
        ablate((cur,tuned)[chosen][0],eps[chosen],mode,[42,43])


if __name__=='__main__':
    main()
