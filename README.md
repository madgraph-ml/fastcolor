<h1 align="center">MadRecolor</h1>

<h2 align="center">Machine-Learned Leading-Color Amplitude Reweighting for MadGraph</h2>

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0-DC583A.svg?style=flat&logo=pytorch"></a>
</p>

**MadRecolor** is a Python-based tool designed to perform **leading-color amplitude reweighting** using machine learning techniques within the **MadGraph** framework. It enables efficient reweighting of matrix elements by leveraging neural networks to approximate color weight factors, improving computational efficiency in high-energy physics simulations.

## Installation
To install **MadRecolor**, clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/madrecolor.git
# then install in dev mode
cd madnis
pip install --editable .
```

## Dependencies
- Python 3.x
- NumPy
- PyTorch
- MadGraph5_aMC@NLO
