This repository contains our work<br />
**Graph Neural Network for Distributed Beamforming and Power Control in Massive URLLC Networks**, which is accepted by the TWC (early access).<br />

**For any reproduce, further research or development, please kindly cite our paper**<br />
@ARTICLE{G4U,<br />
  author={Gu, Yifan and She, Changyang and Bi, Suzhi and Quan, Zhi and Vucetic, Branka},<br />
  journal={IEEE Transactions on Wireless Communications}, 
  title={Graph Neural Network for Distributed Beamforming and Power Control in Massive URLLC Networks},<br /> 
  year={2024},<br />
  volume={},<br />
  number={},<br />
  pages={}, <br />
  note={early access},<br />
  }<br />

**Instructions:**<br />
1. Simulation for GNN, WMMSE and EPA policies can be found in **GNN and WMMSE and EPA.py**.<br />
2. Simulation for the proposed G4U can be found in **G4U.py**.<br />
3. Simulation for the proposed PG4U can be found in **PG4U.py**.<br />
4. Note that we have developed a loss function for the training of URLLC networks. If one want to compare it with the utility function-based one, comment out line 186-189, and use line 192-197 in **GNN and WMMSE and EPA.py** for training. In addition, one may use other loss functions for training, such as error probability-based ones, but they may not achieve a good performance. Similar comments also apply for **G4U.py** and **PG4U.py**.<br />
   
We thank the works "Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and Theoretical Analysis" and "Spatial Deep Learning for Wireless Scheduling" for their source codes in creating this repository.
