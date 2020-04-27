# CSCI 473 - Human-Centered Robotics Project 3: Robot Understanding of Human Behaviors Using Skeleton-Based Representations
This repository was created by Tyler Zudans for Project 3 of CSCI 473. This repository has a directory called dataset which contains text files detailing a human's joint positions in 3D space from a kinect camera. The repository is structured as follows:

**D2 Results**

*RAD Algorithm Implementation*
* Accuracy = 56.25%
* C = 2
* Gamma = 0.5
* 17 bins used

*HJDP (Custom) Algorithm Implementation*
* Accuracy = 68.75%
* C = 2
* Gamma = 0.5
* 18 bins used

Prediction files exist in the representations folder with the extenstion .predict

**Directories**

1. *dataset* - contains testing and training raw data of skeleton representations pulled from a connected structured light depth camera
1. *representations* - contains compressed/custom files (such as rad_d1.t) built from histogram representations of the data in *dataset*. It also contains the *prediction files* 

**Scripts**

1. **|P3-D1|** (Deprecated) The python script [skeleton_representation.py](https://github.com/tylerzudans/CSCI473-SkeletonRepresentation/blob/master/skeleton_representation.py) iterates through the dataset and converts them into a training and test file (e.g. rad_d1.t) for the RAD and HJDP compressions of the dataset as detailed in the assignment.
1. **|P3-D2|** The python script [skeleton_representation_libsvm_format.py](https://github.com/tylerzudans/CSCI473-SkeletonRepresentation/blob/master/skeleton_representation_libsvm_format.py) iterates through the dataset and converts them into a training and test file (e.g. rad_d1.t) for the RAD and HJDP compressions of the dataset as detailed in the assignment. This script converts differently from the last one in that is is LIBSVM compatible. After converting, this script will use the traing data to train a support vector machine (SVM) and tests its accuracy. Finally it writes its label predictions to a file with the .prediction extension in the representation directory.

**Installation**:

From a linux terminal:
1. Clone this repository
   1. $ `git clone https://github.com/tylerzudans/CSCI473-SkeletonRepresentation.git`
1. Install **numpy** and **os** with pip for python 3
   1. $ `pip3 install numpy` or $ `python3 -m pip install numpy`
   1. $ `pip3 install os` or $ `python3 -m pip install os`
   1. $ `pip3 install libsvm` or $ `python3 -m pip install libsvm`

**Run**:

(From linux terminal at this cloned directory)
1. This script will create rad, rad.t, hjdp, and hjpd.t libsvm data representation files, train their respective support vector machines, and test their accuracy. *.predict* files will be generated in the representations folder.
   1. $ `python3 skeleton_representation.py`
1. (Deprecated) To create rad, rad.t, hjdp, and hjpd.t files for part 1 run. These files will appear in the **representations** directory.
   1. $ `python3 skeleton_representation.py`

**Algorithms**

Compression algorithms built according to instructions from the P3-D1 instruction. Histograms built using **numpy**
1. *RAD* - RAD using neck, both wrists, and both ankles, compressed to 5 bin histograms, with little outlier removal
1. *HJDP (a.k.a. Custom)* - HJDP with all joints, compressed to 5 bin histograms, with little outlier removal



**Tutorials Used**

1. 
1.
