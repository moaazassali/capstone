# CAPSTONE REPO
Umer Bin Liaqat (ubl@nyu.edu)
Moaaz Assali (ma5679@nyu.edu)
Fatema Alzaabi (@nyu.edu)

## Note
For more information refer to the DROID-SLAM repo and instructions https://github.com/princeton-vl/DROID-SLAM

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
