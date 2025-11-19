# Coding along "Cuda by examples" book from Sanders

## Run example
```bash
nvcc -arch=sm_86 hash_cpu.cu -o a && ./a
```

## For bitmap required examples
```bash
sudo apt install freeglut3-dev
nvcc ray_tracing.cu -o a.out -lglut -lGLU -lGL -lm && ./a.out
```

## For ripple example
```bash
sudo apt install freeglut3-dev libglu1-mesa-dev mesa-common-dev
nvcc -arch=sm_86 heat.cu -o a.out -lglut -lGLU -lGL -lm && ./a.out
```
