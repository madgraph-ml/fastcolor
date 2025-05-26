# Analytic Amplitude Function

## Installation

### Compile the library
In order to use the analytic function, you first need to compile the library. For this do
```bash
cd cpp/src
make clean
make
cd ../SubProcesses/P1_Sigma_sm_gg_gggg/
make mg5_vectorized.so
```

### Install `madspace` (only needed to run the test file)
Then you also need to install the madspace package:
```bash
# clone the repository (maybe not within here, but as you want)
git clone https://github.com/madgraph-ml/MadSpace.git
# then install
cd MadSpace
pip install .
```
This is not necessarily required to evaluate the amplitude, but it is needed to run the test code. It also provides a custom phase-space sampler with full control over the phase-space weight.

## Run the test code

Then you can see an example file of how to evaluate the amplitude
in `evaluate_m2.py` and you can also simply run it by
```bash
python evaluate_m2.py
```
If this runs through, everything is working well.
