# Implementation of Quantum Volume test and its parity preserving modifications

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Notebook Descriptions](#notebook-descriptions)
5. [License](#license)

## Introduction

This repository contains an implementation of the standard Quantum Volume benchmark and two parity preserving modifications proposed in preprint … . The quantum circuits are constructed using Qiskit and can be executed either on an IBM quantum computer or its simulators. For a detailed description of the abovementioned tests see the preprint.



## Setup

To use this repository, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/RafalBistron/Quantum_Volume
   ```

2. Install the required dependencies. It is recommended to create a virtual environment:

   ```bash
   cd Quantum_Volume
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Run jupyter notebook:

   ```bash
   jupyter notebook
   ```
   

## Notebook Description

QV_calculation
   - Generate circuits to calculate standard Quantum Volume together with its modifications  
   - Run circuits on an IBM quantum computer or its simulator
   - Process the results to derive heavy output frequencies
   - Plot heavy output frequencies



## License

This repository is licensed under the MIT License. Feel free to use the code and notebook for your educational and personal projects.

---