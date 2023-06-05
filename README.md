# Conformal Prediction with Partially Labeled Data
This repo has the code for the paper: [Conformal Prediction with Partially Labeled Data](https://arxiv.org) by Alireza Javanmardi, Yusuf Sale, Paul Hofman, and Eyke Hüllermeier. This paper will be appear in the 12th Symposium on Conformal and Probabilistic Prediction with Applications (COPA 2023).

## Setup
The file `requirements.txt` has all the required pip packages. The code can be run using the `main.py` file. An example can be found below
```
python3 main.py -ds mnist -e 100 -noise random -p 1.0 -q 0.1
```

## Citation
If you use this code, please cite the paper
```
@misc{javanmardi2023conformal,
      title={Conformal Prediction with Partially Labeled Data}, 
      author={Alireza Javanmardi and Yusuf Sale and Paul Hofman and Eyke Hüllermeier},
      year={2023},
      eprint={2306.01191},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
