"""Unlabeled calibration-error transforms, fixed before new test evaluation."""
import numpy as np
VARIANTS=('field','flat','mean_scale','median_scale','tail','tail_flat','tail_max','tail_corr')

class FieldScorer:
    @staticmethod
    def validate(x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all() or (x<0).any(): raise ValueError('Finite nonnegative NxG errors required')
        return x

    def fit(self, calibration, widths):
        x=self.validate(calibration);self.widths=np.asarray(widths,dtype=np.float64)
        if self.widths.shape!=(x.shape[1],) or not np.isfinite(self.widths).all() or (self.widths<=0).any():raise ValueError('Invalid field widths')
        self.sorted=np.sort(x,axis=0);mean=x.mean(0);median=np.median(x,axis=0)
        self.mean=np.where(mean>1e-8,mean,1.)
        self.median=np.where(median>1e-8,median,self.mean)
        z=self.evidence(x);g=x.shape[1]
        if g==1:corr=np.zeros((1,1))
        else:
            centered=z-z.mean(0);cov=centered.T@centered;norm=np.sqrt(np.diag(cov));denom=norm[:,None]*norm[None,:]
            corr=np.divide(cov,denom,out=np.zeros_like(cov),where=denom>0)
        np.fill_diagonal(corr,0);w=1/(1+np.clip(corr,0,1).sum(1));self.corr_weights=w/w.sum()
        return self

    def evidence(self,x):
        x=self.validate(x)
        if x.shape[1]!=self.sorted.shape[1]:raise ValueError('Field mismatch')
        n=len(self.sorted)
        p=np.column_stack([(1+n-np.searchsorted(self.sorted[:,g],x[:,g],side='left'))/(n+1) for g in range(x.shape[1])])
        return -np.log(p)

    def all_scores(self,x):
        x=self.validate(x);z=self.evidence(x)
        return dict(field=x.mean(1),flat=np.average(x,axis=1,weights=self.widths),mean_scale=(x/self.mean).mean(1),median_scale=(x/self.median).mean(1),tail=z.mean(1),tail_flat=np.average(z,axis=1,weights=self.widths),tail_max=z.max(1),tail_corr=z@self.corr_weights)

    def save(self,path):np.savez_compressed(path,sorted=self.sorted,widths=self.widths,mean=self.mean,median=self.median,corr_weights=self.corr_weights)

    @classmethod
    def load(cls,path):
        o=cls()
        with np.load(path,allow_pickle=False) as z:
            for k in z.files:setattr(o,k,z[k])
        return o


class SkillScorer:
    def fit(self,train,cal,errors,field_groups):
        self.groups=field_groups;self.center=np.median(np.asarray(train,dtype='f8'),axis=0)
        baseline=(np.asarray(cal,dtype='f8')-self.center)**2
        self.null_error=np.array([baseline[:,g].mean() for g in self.groups]);self.model_error=np.asarray(errors).mean(0)
        if len(errors)!=len(cal) or not np.isfinite(self.model_error).all() or (self.model_error<0).any():raise ValueError('Invalid errors')
        self.widths=np.array([len(g) for g in self.groups],dtype='f8');self.skill=np.clip(1-self.model_error/np.maximum(self.null_error,1e-8),0,1)
        def norm(w):return w/w.sum() if w.sum()>0 else self.widths/self.widths.sum()
        self.weights=norm(self.widths*(.25+.75*self.skill));self.equal=norm(.25+.75*self.skill);self.hard=norm(self.widths*self.skill)
        self.tail=FieldScorer().fit(errors,self.widths);return self
    def all_scores(self,errors):
        e=np.asarray(errors,dtype='f8');z=self.tail.evidence(e)
        return dict(field=e.mean(1),flat=np.average(e,axis=1,weights=self.widths),tail=z.mean(1),tail_flat=np.average(z,axis=1,weights=self.widths),skill_tail=z@self.weights,skill_raw=e@self.weights,skill_equal_tail=z@self.equal,skill_hard_tail=z@self.hard)
    def save(self,path):np.savez_compressed(path,center=self.center,null_error=self.null_error,model_error=self.model_error,widths=self.widths,skill=self.skill,weights=self.weights,equal=self.equal,hard=self.hard,sorted=self.tail.sorted)

    @classmethod
    def load(cls,path):
        obj=cls()
        with np.load(path,allow_pickle=False) as z:
            for name in ['center','null_error','model_error','widths','skill','weights','equal','hard']:setattr(obj,name,z[name])
            obj.tail=FieldScorer();obj.tail.sorted=z['sorted'];obj.tail.widths=obj.widths
        return obj
