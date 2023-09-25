# JAX OpenGL Height-Field Water Simulator

[![Simulation YouTube Video](images/simulation.jpeg)](https://youtu.be/Vrl7CklleO4?si=gBdu8EDwAdmfNjJm)

This is a simple height-field water simulator written in Python using `JAX` and `OpenGL`. It is based on the awesome work of [Matthias](https://www.youtube.com/watch?v=hswBi5wcqAA) which was implemented using `three.js`.

The main differences:
* Python
* Uses JAX so simulator code can run on CPU, GPU, METAL and TPU (TPU untested)
* Uses pure OpenGL with no external libraries

This application has been tested with Python 3.10 on CPU and METAL GPU on MacOS. It should work on any platform that supports `JAX` and `OpenGL`. On my MacBook Pro (M1) it runs at 350fps with 101x101 grid when using JAX CPU.

## Run Instructions
First make sure you have Python.
```
git clone https://github.com/cemlyn007/water-simulator
cd water-simulator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --n 101
```
