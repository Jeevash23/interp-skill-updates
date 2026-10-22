"""One frozen near-linear diagnostic; reuse the original experiment engine."""
import argparse
from pathlib import Path
import shutil
import study as s

original_network=s.network


def network(seed,architecture,width):
    if architecture!='leaky':return original_network(seed,architecture,width)
    s.torch.manual_seed(seed)
    return s.torch.nn.Sequential(s.torch.nn.Linear(16,width,bias=False),s.torch.nn.LeakyReLU(.9),
                                 s.torch.nn.Linear(width,1,bias=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--tasks',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.mode=='prepare':s.prepare(a.config,a.output)
    else:
        s.network=network
        # Supplement the engine's snapshot immediately when its output is first written.
        original_write=s.write
        def write(path,value):
            if Path(path)==a.output.resolve()/'manifest.json':
                for source in [Path(__file__).resolve(),s.ROOT/'protocol/AMENDMENT.md']:
                    rel=source.relative_to(s.ROOT);dest=a.output/'snapshot'/rel
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
                    value['source_hashes'][str(rel)]=s.sha(source)
            return original_write(path,value)
        s.write=write
        s.execute(a.config,a.tasks,a.output)
