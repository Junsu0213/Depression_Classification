# Depression Classification

Here is a simple repo for classifying MDD and normal controls via resting-state EEG signals.


<div align=center>

![mdd pipeline](https://github.com/Junsu0213/Depression_Classification/assets/128777619/50d03927-aeec-42f7-86ed-b4c7c6acee64)

**Flow chart**

</div>

## 1. Installation
#### Environment
* Python == 2.8.2
* MNE == 1.1.0
* mne-connectivity == 0.3
* scipy == 1.7.3
* sklearn == 0.0
* shap == 0.41.0

## 2. Directory structure
```bash
├── Config
│   └── config.py
├── Evaluation
│   └── k_fold.py
├── Loader
│   └── load.py
├── Model
│   ├── MLClassifier.py
│   └── ml_grid.py
├── main.py
└── requirements.txt
```

## 3. Dataset

#### [PRED+CT MDD dataset](http://predict.cs.unm.edu/)
* 116 subjects (MDD: 73, Healthy control: 43)
* Resting state EEG data: eyes open (EO), eyes closed (EC)
