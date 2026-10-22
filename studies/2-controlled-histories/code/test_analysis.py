"""Analytic checks of family aggregation, intervals and the nonlinear amendment."""
import unittest
import numpy as np
from scipy.stats import t
import analyze as a
import study as s
import amended_study as amended
s.initialize(True)


class AnalysisChecks(unittest.TestCase):
    def test_family_aggregation_not_pseudoreplication(self):
        rows=[]
        for seed,delta in [(1,-.1),(2,.1),(3,.3)]:
            for init in [4,5]:
                for branch in ['f','g']:
                    for hist in ['independent','joint']:
                        rows.append(dict(task_seed=seed,model_seed=init,branch=branch,architecture='linear',
                            history=hist,arm='unrestricted',stable=.5+(delta if hist=='independent' else 0)))
        out=a.paired(rows,'linear','unrestricted','stable',.05/3)
        self.assertEqual(out['n'],3)
        self.assertAlmostEqual(out['mean'],.1)
        self.assertAlmostEqual(out['sd'],.2)
        self.assertAlmostEqual(out['high'],.1+t.ppf(1-.05/6,2)*.2/np.sqrt(3))

    def test_qualification_all_endpoints(self):
        c=s.read(s.ROOT/'protocol/development.json');e=dict(f=.01,g=.01,combined_seen=.01,combined_unseen=.01)
        self.assertTrue(a.qualified(e,c))
        for key in e:self.assertFalse(a.qualified(dict(e,**{key:.11}),c))

    def test_amended_network(self):
        torch=s.torch;net=amended.network(1,'leaky',32);task=s.make_task(60021)
        _,v=s.rows_and_inputs();w=np.r_[task['f'][0],task['g'][0]]
        with torch.no_grad():
            net[0].weight.copy_(torch.tensor(np.r_[np.eye(16),-np.eye(16)]))
            net[2].weight.copy_(torch.tensor(np.r_[w,-w][None,:]/1.9))
        np.testing.assert_allclose(s.pred(net,torch.tensor(v)),s.labels(task,'00'),atol=1e-12)
        self.assertEqual(sum(p.numel() for p in net.parameters()),544)


if __name__=='__main__':unittest.main(verbosity=2)
