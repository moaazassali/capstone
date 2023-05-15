# CAPSTONE REPO
Umer Bin Liaqat (ubl203@nyu.edu)

Moaaz Assali (ma5679@nyu.edu)

Fatema Alzaabi (fya210@nyu.edu)

## Some Info:
- We use DROID-SLAM, so for more information refer to the DROID-SLAM repo and instructions https://github.com/princeton-vl/DROID-SLAM
- We made some changes in several of the original files to get the model to work on our system without errors
- We used Python 3.9.13

## Code files that we added
- droid_slam_node.py: Connecting the model with ROS by creating a node that receives image input from a specific ROS topic
- img_receiver.py: Receive images using a socket connection rather than ROS. We didn't complete this but it might be faster than using ROS especially when dealing with Unity, where ROS might be an unnecessary overhead in this case.

## Getting Started
1. Clone the repo
```Bash
git clone https://github.com/moaazassali/capstone.git
```

2. Creating a new anaconda environment using the provided requirements.txt file
```Bash
conda create --name droidenv --file requirements.txt
conda activate droidenv
pip install evo --upgrade --no-binary evo
pip install gdown
```

3. Compile the extensions (takes about 10 minutes)
```Bash
python setup.py install
```

4. Run demo
```Bash
./tools/download_sample_data.sh
python demo.py --imagedir=data/abandonedfactory --calib=calib/tartan.txt --stride=2
```
