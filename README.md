# Fundamental bounds of Wavefront Shaping of Spatially Entangled Photons

### Introduction

This repository contains the Python code implementation for the following publication:

> Ronen Shekel, Sébastien M. Popoff and Yaron Bromberg. Fundamental bounds of Wavefront Shaping of Spatially Entangled Photons, 2025. 

**Abstract**: 

Wavefront shaping enables control of classical light through scattering media. Extending these techniques to spatially entangled photons promises new quantum applications, but the fundamental limits, especially when both photons scatter, remain unclear. Here, we theoretically and numerically investigate the enhancement of biphoton correlations through thick scattering media. We analyze configurations where a spatial light modulator shapes one or both photons, either before or after the medium, and show that the optimal enhancement differs fundamentally from classical expectations. For a system with $N$ modes, we show that shaping one photon yields the classical enhancement $\eta \approx (\pi/4)N$, while shaping both photons before the medium reduces it to $\eta \approx (\pi/4)^2N$. However, in some symmetric detection schemes, when both photons are measured at the same detection mode, perfect correlations are restored with $\eta \approx N$, resembling digital optical phase conjugation. Additionally, shaping both photons after the medium leads to a complex, NP-hard-like optimization problem, yet achieves superior enhancements, with coincidence rates boosted up to $\eta \approx 4.6N$. These results reveal unique quantum effects in complex media and identify strategies for quantum imaging and communication through scattering environments.


### Project Structure
- `qwfs_simulation.py`: The main simulation code.
- `qwfs_result.py`: Contains the QWFSResult class, that holds the data, and has saveto and loadfrom methods.
- `generate_results.ipynb`: A notebook for generating the results shown in the paper.
- `generate_results.ipynb`: A notebook using the output of `generate_results.ipynb` to generate the figures that are shown in the paper. 
- `data\`: directory saving previously run results

### Important note
The naming of the configurations is different in the paper and in the code:

| Paper                                 | code           | 
|---------------------------------------|----------------|
| 1P-S                                  | SLM1           | 
| 2P-IS                                 | SLM2           | 
| 2P-IS(OPC)                            | SLM2-same-mode | 
| 2P-DS                                 | SLM3           | 
| 2P-DS(OPC)                            | SLM3-same-mdoe |
| 2P-IS with crystal not in image plane | SLM5           |


## Citation
Please cite the publication if you use this code or data in your research.
