# Automated Website Fingerprinting using Deep Learning
_CSE 534 - Fundamentals of Computer Networks_

## Reference Paper
Rimmer, V., Preuveneers, D., Juarez, M., Goethem, T. V., &amp; Joosen, W. (2018). Automated website fingerprinting through Deep Learning. Proceedings 2018 Network and Distributed System Security Symposium. https://doi.org/10.14722/ndss.2018.23105

## Webpage by the authors of the above paper
https://distrinet.cs.kuleuven.be/software/tor-wf-dl/

## Dataset
The dataset needs to be downloaded from https://github.com/DistriNet/DLWF.  
For our implementation, we have used the Closed World dataset of 100 websites consisting of 2500 traces each.  
The dataset can be downloaded from https://distrinet.cs.kuleuven.be/software/tor-wf-dl/files/tor_100w_2500tr.npz.  
  
Once the dataset is download, unzip the `.npz` file and copy the extracted files (`data.npy` and `labels.npy`) in the `./data/` folder.

