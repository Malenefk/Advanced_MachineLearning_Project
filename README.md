# Advanced Machine Learning Course FYS5429
**Group Project by: Malene Fjeldsrud Karlsen & Olaf Wojcinski**

This GitHub site is dedicated to the Machine Learning FYS5429 project. This Readme contains information on the folder structure of the GitHub page, the necessary information for reproducing results, and the paper abstract. 

# Upload pic here for visuallllziing (jeg fikser)

**Abstract**

Predicting turbulence is of great importance for both operational use and climate research. Turbulence plays a key role in boundary layer dynamics near the land and ocean surface, and is the reason behind much of the short-term weather variability we observe. The non-linear dynamics associated with turbulence make it difficult to model accurately, and current approaches rely on parameterizations that are computationally expensive and time-consuming to solve. By testing four deep learning models, a convolutional neural network (UNet), a Fourier neural operator (FNO), and two spectral graph-based convolutional networks, this study examines whether neural networks can emulate the associated turbulence dynamics to a satisfactory extent. All four models are evaluated across five loss functions on an offline, low-resolution quasi-geostrophic dataset. 

The FNO outperforms the other models with the lowest accumulated RMSE score and a physically consistent kinetic energy spectrum. The UNet retains high-frequency variance in the forecast field and is least affected by spatial smoothing. The GCN-based models are outperformed by both, likely due to an overly complex graph architecture relative to the regular grid structure of the input data. Amplitude collapse is evident across all four models, with RMSE scores accumulating across forecast steps as predictions progressively lose spatial frequency and spatial structure.

This report highlights the strengths and weaknesses of different model architectures in the context of offline turbulence emulation for five loss functions. The results suggest that architectural stability dominates over loss function in terms of model performance, which raises questions for future work, including the use of alternative loss functions and reduced temporal sampling. 

**Folder structure**

- **Code**
  - *CNN & FNO:*
    Contains all the utilities and code for training the quasi-geostrophic turbulence model for the Convolutional Neural Network and for the Fourier Neural Operator.
    **Her skriver Olaf om sin egen allerede kutta fil**
  - *GCN:*
    Contains all utilities and code for training with a Weighted Chebyshev model, a regular Chebyshev model, a deep architectural Chebyshev model, and a standard Graph Convolutional Network model.
    *train.nc* is the NetCdf file used for training the models.
    To run the scripts, the Python script "run_multi_variable.py" must be sent in. The script calls on all other scripts used for training, such as loss functions, model architecture, and more. To select hyperparameters, the values in the SWEEP can be changed accordingly.
    The supported models to run using this script are:  'ChebGCN', 'GCN', 'DeepChebGCN' and 'WeightedChebGCN'
    The configuration file contains information on the parameters that must be specified for the code to run successfully, while the SWEEP parameters, such as penalties, can be changed to zero or, in some cases, False/True.
    **NOTE:** To run the run_multi_variable script, the other scripts within Code -> GCN -> must be downloaded in the same folder to successfully import the scripts. 
  - *Validation*
    Contains the script to run inference and to plot the results. The KE notebook includes a theoretical view of how the kinetic energy spectrum is calculated.
    
- **Report**
  This section contains the report in a PDF file.

- **Report**
  The Figures folder contains all figures that have been plotted and that are used in the report. 
  
- **LLM**
  A declaration of Artificial Intelligence is given. 
