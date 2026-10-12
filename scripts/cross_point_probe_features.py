"""Independent readout features. GT future arrays never enter this API."""
import numpy as np


def features(observed_xyz, baseline_future, latent_future, query_uv, neighbors=True, k=8):
    observed=np.asarray(observed_xyz,dtype=np.float64)
    baseline=np.asarray(baseline_future,dtype=np.float64)
    latent=np.asarray(latent_future,dtype=np.float64)
    uv=np.asarray(query_uv,dtype=np.float64)
    if observed.ndim!=3 or observed.shape[-1]!=3 or baseline.ndim!=3 or baseline.shape[-1]!=3:
        raise ValueError('Expected time,query,xyz')
    n=observed.shape[1]
    if n<=k or baseline.shape[1]!=n or latent.shape[:2]!=baseline.shape[:2] or uv.shape!=(n,2):
        raise ValueError('Mismatched query identity/count')
    if not all(np.isfinite(x).all() for x in [observed,baseline,latent,uv]):
        raise ValueError('Nonfinite observed/model inputs')
    origin=observed[-1].mean(0)
    scale=float(np.sqrt(np.mean(np.sum((observed[-1]-origin)**2,axis=-1))))
    scale=max(scale,1e-6)
    own=(observed-origin)/scale
    future=(baseline-origin)/scale
    shared=np.concatenate([own.mean(1).ravel(),future.mean(1).ravel(),latent.mean(1).ravel()])
    own_block=np.concatenate([own.transpose(1,0,2).reshape(n,-1),
        future.transpose(1,0,2).reshape(n,-1),latent.transpose(1,0,2).reshape(n,-1),
        np.broadcast_to(shared,(n,len(shared)))],axis=-1)
    d=np.linalg.norm(uv[:,None]-uv[None,:],axis=-1); np.fill_diagonal(d,np.inf)
    index=np.argsort(d,axis=1,kind='stable')[:,:k]
    # Relative to the target's last observed point, with full temporal history.
    extra=(own[:,index]-own[-1][None,:,None,:]).transpose(1,2,0,3).reshape(n,-1)
    extra=np.concatenate([extra,np.ones((n,1))],axis=-1)
    if not neighbors: extra=np.zeros_like(extra)
    return np.concatenate([own_block,extra],axis=-1),{'origin':origin,'scale':scale,
        'neighbors':index,'self_feature_width':own_block.shape[1],'neighbor_feature_width':extra.shape[1]}


def training_standardizer(train_features):
    """Training split only; explicitly keep constant/masked channels harmless."""
    x=np.concatenate(train_features,axis=0)
    mean=x.mean(0); std=x.std(0); std=np.where(std<1e-8,1.,std)
    return mean,std


def standardize(x,mean,std):
    return (np.asarray(x)-mean)/std


def ridge_fit(x,y,alpha):
    """No future information is passed as a feature; y is training labels only."""
    xm=x.mean(0); ym=y.mean(0); xc=x-xm; yc=y-ym
    penalty=np.eye(x.shape[1])*alpha*len(x)
    coefficient=np.linalg.solve(xc.T@xc+penalty,xc.T@yc)
    return coefficient,ym-xm@coefficient


def check_cache(cache,observed_frames=7,total_frames=22):
    required=['observed_xyz','baseline_future','latent_future','query_uv','timestamps_s',
        'coordinate_frame','future_rgb_supplied','gt_input_supplied']
    for key in required:
        if key not in cache: raise ValueError('Missing model cache field: '+key)
    if bool(cache['future_rgb_supplied']) or bool(cache['gt_input_supplied']):
        raise ValueError('Future/GT input leakage')
    if str(cache['coordinate_frame'])!='last_observed_camera': raise ValueError('Coordinate mismatch')
    if len(cache['observed_xyz'])!=observed_frames or len(cache['baseline_future'])!=total_frames-observed_frames:
        raise ValueError('Wrong temporal boundary')
    timestamps=np.asarray(cache['timestamps_s'])
    if len(timestamps)!=total_frames or not np.all(np.diff(timestamps)>0): raise ValueError('Invalid timestamps')
    return features(cache['observed_xyz'],cache['baseline_future'],cache['latent_future'],cache['query_uv'])
