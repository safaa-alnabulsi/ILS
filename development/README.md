# Real-time Image Smoothing via Iterative Least Squares 

Python implementation of the following paper: "Real-time Image Smoothing via Iterative Least Squares". Wei Liu, Pingping Zhang, Xiaolin Huang, Jie Yang, Chunhua Shen, Ian Reid. ACM Transactions on Graphics (TOG), 39(3), 1-24.

## Requirements

- Python 3.6
- conda
  - macOS
  ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
     bash ~/miniconda.sh -b -p $HOME/miniconda
   ```


## Usage

Create a conda environment and install the python required libraries by running the following commands

```bash
conda create -n SeminarGraphicsB-ILS-py36 python=3.6 -y
conda activate SeminarGraphicsB-ILS-py36
pip install -r requirements.txt
```

To start jupyter notbook, run:

```bash
jupyter notebook
```

## References
- https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
- https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
- https://jupyter.org/install.html
- https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
- https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_filtering/py_filtering.html#d-convolution-image-filtering
- https://hgomersall.github.io/pyFFTW/pyfftw/interfaces/interfaces.html
