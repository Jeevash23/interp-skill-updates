"""Numerical checks using fixtures outside development and confirmation seeds."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import study as s
s.initialize()
np,torch=s.np,s.torch
C=s.read(s.ROOT/'protocol/development.json')


class StudyChecks(unittest.TestCase):
    def test_tasks_and_matching(self):
        for seed in range(60001,60021):
            task=s.make_task(seed);check=s.check_design(task,C)
            self.assertEqual(check['training_examples'],64)
            self.assertEqual(check['withheld_examples'],16)
            for env in ['00','13','40','04']:
                y=s.labels(task,env);f,g=map(int,env)
                np.testing.assert_allclose(y[16:].reshape(8,8),np.array(task['f'][f])[:,None]+np.array(task['g'][g])[None,:])
            self.assertEqual(check['changed_function_counts']['independent'][1],22)
            self.assertEqual(check['changed_function_counts']['joint'][1],4)

    def test_shared_initialization(self):
        for arch in C['architectures']:
            self.assertEqual(s.model_hash(s.network(123,arch,32)),s.model_hash(s.network(123,arch,32)))

    def test_masks(self):
        for branch in ['f','g']:
            for arm in C['arms']:
                masks=s.mask_for(arm,60001,123,branch,32)
                again=s.mask_for(arm,60001,123,branch,32)
                for a,b in zip(masks,again):self.assertTrue(torch.equal(a,b))
                n=sum(int((m==0).sum()) for m in masks)
                self.assertEqual(n,512 if arm=='readout_only' else 32 if arm=='freeze_readout' or arm.startswith('random_') else 0)
                if arm.startswith('random_'):
                    other=masks[0][:,8:] if branch=='f' else masks[0][:,:8]
                    self.assertTrue(torch.equal(other,torch.ones_like(other)))

    def test_exact_functions_and_interventions(self):
        task=s.make_task(60001);_,v=s.rows_and_inputs();x=torch.tensor(v)
        for arch in C['architectures']:
            model=s.network(123,arch,32);f=np.array(task['f'][0]);g=np.array(task['g'][0]);w=np.r_[f,g]
            with torch.no_grad():
                model[0].weight.copy_(torch.tensor(np.r_[np.eye(16),-np.eye(16)]))
                model[2].weight.copy_(torch.tensor(np.r_[w,-w][None,:]/(2 if arch=='linear' else 1)))
            np.testing.assert_allclose(s.pred(model,x),s.labels(task,'00'),atol=1e-12)
            cfg=dict(C,update_steps=3)
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);s.updates(task,123,model,cfg,folder,x)
                out=np.load(folder/'updates.npz');ends=np.load(folder/'endpoints.npz');meta=s.read(folder/'updates.json')
                for bi,branch in enumerate(['f','g']):
                    stable=slice(8,16) if branch=='f' else slice(0,8)
                    for ai,arm in enumerate(C['arms']):
                        a=ends[f'{branch}__{arm}__0.weight'];b=ends[f'{branch}__{arm}__2.weight']
                        h=v@a.T
                        if arch=='relu':h=np.maximum(0,h)
                        np.testing.assert_allclose((h@b.T).ravel(),out['predictions'][bi,ai,-1],atol=1e-12)
                        if arm in ['freeze_readout','no_update']:
                            np.testing.assert_array_equal(out['predictions'][bi,ai,:,stable],np.broadcast_to(s.labels(task,'00')[stable],(4,8)))
                        if arm=='normalized':np.testing.assert_allclose(out['step_length'][bi,ai,1:],.01,atol=1e-12)
                        if arm=='old_primitive':np.testing.assert_allclose(out['predictions'][bi,ai],np.broadcast_to(s.labels(task,'00'),(4,80)),atol=1e-12)
                self.assertLess(max(t['readout_identity_max_error'] for t in meta['trajectories']),1e-10)


if __name__=='__main__':unittest.main(verbosity=2)
