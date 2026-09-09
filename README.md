This repository is part of the following publication:

PccDiffuser: Multi-solution Motion Planning for Continuum Robots

Link: 

---

### Generate dataset (smoke test)
```bash
python -u scripts/generate_dataset.py --output-dir data/set_smoke --smoke --overwrite --device cuda:0
```

### Generate dataset
```bash
python -u scripts/generate_dataset.py --output-dir data/set_v1 --target-samples 10000 --start-count 5 --terminal-count 10 --sample-batch-size 32 --obstacle-counts 0 1 2 4 --seed 0 --device cuda:0
```

### Train diffusion model
```bash
python -u scripts/train_model.py --config configs/train.yaml --dataset data/set_v1 --output-dir runs --device cuda:0
```

### Evaluate diffusion model
```bash
python -u scripts/evaluate_model.py --output-dir runs/set_v1 --dataset data/set_v1 --checkpoint runs/set_v1/final.pt --device cuda:0
```

### Execute all (generate, train, evaluate)
```bash
python -u scripts/execute_all.py --dataset data/set_v1 --config configs/train.yaml --output-dir runs --target-samples 10000 --start-count 5 --terminal-count 10 --obstacle-counts 0 1 2 4 --seed 0 --device cuda:0
```

### Evaluate benchmarks
```bash
python -u scripts/evaluate_benchmarks.py --output-dir runs --dataset data/set_v1 --algorithm c-rrt c-rrt-star w-rrt w-rrt-star repulsion --device cpu
```
