This is a sample repository for Visual Odometry between subsequent images. The project contains the following structure:
```
.
├── config.yaml
├── data
│   ├── 000000.png
│   ├── 000001.png
│          .
           .
│   └── 000011.png
├── Part1.py
├── README.md
└── requiremts.txt
```
Before setting up create a virtual environmnet in python and install the requirements file. 
The code was developed in Python 3.12.3, and hence the virtual environmnet should be of the same as are the requirements file.
```bash
python3 -m venv venv
. venv/bin/activate
cd zipped_file_submission_path
python Part1.py
```

The Part1.py includes the solution to the question 1-3 in part 1 on the given sample subset of KITTI dataset. Please look at the 
config.yaml file for the parameters like camera matrix and dataset location etc, used in the project.