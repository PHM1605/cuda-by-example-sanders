# Coding along "Cuda by examples" book from Sanders

## Run example
```bash
nvcc -arch=sm_86 add_kernel.cu -o a && ./a
```

## For bitmap required examples
```bash
sudo apt install freeglut3-dev
nvcc julia_set_cpu.cu -o a -lglut -lGLU -lGL -lm && ./a
```

## For ripple example
```bash
sudo apt install freeglut3-dev libglu1-mesa-dev mesa-common-dev
nvcc -arch=sm_86 ripple.cu -o ripple -lglut -lGLU -lGL -lm && ./ripple
```