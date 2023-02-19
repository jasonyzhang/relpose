# Evaluation

For a trained model, use `relpose/eval/eval_joint.py` for evaluation. For example,
to evaluate using a maximum spanning tree over 5 frames from seen object categories,
use:
```
python -m relpose.eval.eval_joint \
        --checkpoint data/pretrained_co3dv1/checkpoints/ckpt_000400000.pth \
        --num_frames 5 \
        --use_pbar \
        --dataset co3dv1 \
        --categories_type seen \
        --mode mst
```

To simplify evaluation, we provide a script to generate a shell file with the commands
for different numbers of frames, evaluation modes, etc. Please see
`relpose/eval/eval_driver.py` which generate a script `eval_jobs.sh`. You can then run
this script with `sh eval_jobs.sh`.

Note: coordinate ascent must be run after the MST evaluation because it is initialized
from the MST solution.

## Expected evaluation results

These models were retrained and may not match the numbers in the paper. There may also
be some stochasticiy in the runs.

### CO3Dv2

Expected evaluation results (Uniform, seen categories):
```
Sequential  N=3  N=5 N=10 N=20
Acc <15°   0.38 0.36 0.33 0.29
Acc <30°   0.61 0.59 0.57 0.54

MST         N=3  N=5 N=10 N=20
Acc <15°   0.38 0.44 0.46 0.43
Acc <30°   0.61 0.63 0.64 0.61

Coord Asc   N=3  N=5 N=10 N=20
Acc <15°   0.81 0.57 0.56 0.55
Acc <30°   0.91 0.73 0.73 0.72
```


Expected evaluation results (Uniform, unseen categories):
```
Sequential  N=3  N=5 N=10 N=20
Acc <15°   0.28 0.27 0.27 0.24
Acc <30°   0.48 0.46 0.47 0.47

MST         N=3  N=5 N=10 N=20
Acc <15°   0.29 0.32 0.37 0.37
Acc <30°   0.48 0.50 0.52 0.53

Coord Asc   N=3  N=5 N=10 N=20
Acc <15°   0.75 0.49 0.45 0.47
Acc <30°   0.86 0.63 0.62 0.62
```


### CO3Dv1

Expected evaluation results (Uniform, seen categories):
```
Sequential  N=3  N=5 N=10 N=20
Acc <15°   0.31 0.30 0.30 0.28
Acc <30°   0.54 0.51 0.51 0.51

MST         N=3  N=5 N=10 N=20
Acc <15°   0.30 0.33 0.35 0.34
Acc <30°   0.53 0.54 0.55 0.53

Coord Asc   N=3  N=5 N=10 N=20
Acc <15°   0.40 0.39 0.44 0.45
Acc <30°   0.60 0.59 0.63 0.64
```


Expected evaluation results (Uniform, unseen categories):
```
Sequential  N=3  N=5 N=10 N=20
Acc <15°   0.18 0.21 0.23 0.25
Acc <30°   0.39 0.38 0.43 0.46

MST         N=3  N=5 N=10 N=20
Acc <15°   0.19 0.22 0.25 0.27
Acc <30°   0.41 0.42 0.42 0.43

Coord Asc   N=3  N=5 N=10 N=20
Acc <15°   0.29 0.27 0.31 0.34
Acc <30°   0.48 0.48 0.51 0.52
```
