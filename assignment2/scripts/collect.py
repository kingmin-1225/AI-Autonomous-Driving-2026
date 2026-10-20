"""과제 2 collect 실행."""
from pathlib import Path
import argparse
import json
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from rl_workflow import run_driving, train
import yaml

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--max-steps',type=int,default=30000)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--delay',type=float,default=0.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.max_steps<1 or (args.episodes is not None and args.episodes<1) or (args.steps is not None and args.steps<1) or args.delay<0:
        parser.error('steps/episodes must be positive; delay must be nonnegative')
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    result=run_driving(root, config, collect=True, headless=args.headless, episodes=args.episodes, max_steps=args.max_steps)
    if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
