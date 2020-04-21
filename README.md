# CSCI 473 - Human-Centered Robotics Project 3: Robot Understanding of Human Behaviors Using Skeleton-Based Representations
This repository was created by Tyler Zudans for Project 3 of CSCI 473. This repository has a directory called dataset which contains text files detailing a human's joint positions in 3D space from a kinect camera. The repository is structured as follows:

**Directories**

1. *dataset* - contains testing and training raw data of skeleton representations pulled from a connected structured light depth camera
1. *representations* - contains compressed/custom files (such as rad_d1.t) built from histogram representations of the data in *dataset*

**Scripts**

1. **|P3-D1|** The python script [skeleton_representation.py](https://github.com/tylerzudans/CSCI473-SkeletonRepresentation/blob/master/skeleton_representation.py) iterates through the dataset and converts them into a training and test file (e.g. rad_d1.t) for the RAD and HJDP compressions of the dataset as detailed in the assignment.
1. **P3-D2**

**Installation**:
From a linux terminal:
1. Clone this repository
   1. $ `git clone https://github.com/tylerzudans/CSCI473-SkeletonRepresentation.git`
1. Install **numpy** and **os** with pip for python 3
   1. $ `pip3 install numpy` or $ `python3 -m pip install numpy`
   1. $ `pip3 install os` or $ `python3 -m pip install os`

**Run**:
(From linux terminal at this cloned directory)

1. To create rad, rad.t, hjdp, and hjpd.t files for part 2 run. These files will appear in the **representations** directory.
   1. $ `python3 skeleton_representation.py`

**Algorithms**

1. *RAD* -
1. *HJDP* -

**Results**


**Tutorials Used**

1. 
1.
