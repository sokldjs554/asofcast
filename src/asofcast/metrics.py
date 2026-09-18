"""Prediction error and simulated acquisition wait are separate measurements."""
from __future__ import annotations
import numpy as np

def _validated(targets,predictions,steps,waits):
    from asofcast.policy import validate_waits
    y,p,steps=np.asarray(targets,dtype=float),np.asarray(predictions,dtype=float),np.asarray(steps)
    waits=validate_waits(waits)
    if y.ndim!=1 or len(y)==0 or p.shape!=(len(y),len(waits)) or steps.shape!=y.shape: raise ValueError('incompatible metric shapes')
    if not np.isfinite(y).all() or not np.isfinite(p).all(): raise ValueError('metrics require finite predictions and targets')
    if not np.issubdtype(steps.dtype,np.integer) or (steps<0).any() or (steps>=len(waits)).any(): raise ValueError('decision indices must lie within the deadline grid')
    return y,p,steps,waits

def score(targets,predictions,steps,waits)->dict:
    y,p,steps,waits=_validated(targets,predictions,steps,waits); errors=p[np.arange(len(y)),steps]-y; selected_wait=waits[steps]
    return {'n':len(y),'mae':float(np.mean(np.abs(errors))),'rmse':float(np.sqrt(np.mean(errors**2))),
            'mean_wait_seconds':float(np.mean(selected_wait)),'p95_wait_seconds':float(np.quantile(selected_wait,.95)),
            'deadline_violation_rate':float(np.mean(selected_wait>waits[-1])),
            'deadline_scope':'simulated decision grid only; compute/network excluded','step_counts':np.bincount(steps,minlength=len(waits)).tolist()}

def random_matched_score(targets,predictions,learned_steps,waits)->dict:
    y,p,steps,waits=_validated(targets,predictions,learned_steps,waits); probabilities=np.bincount(steps,minlength=len(waits))/len(steps); errors=p-y[:,None]
    return {'n':len(y),'mae':float(np.abs(errors).mean(axis=0)@probabilities),
            'rmse':float(np.sqrt((errors**2).mean(axis=0)@probabilities)),'mean_wait_seconds':float(waits@probabilities),
            'wait_probabilities':probabilities.tolist(),'comparison_scope':'post-hoc exact random-mixture expectation, not deployment'}
