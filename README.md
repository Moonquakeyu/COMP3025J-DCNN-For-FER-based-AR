# COMP3025J-DCNN-For-FER-based-AR  

This repository is for the COMP3025J ARVR case study code, which can run in a Python environment.  

## Model Information  

- The model is in the format of a Jupyter Notebook (`.ipynb` file).   
- Users need to open this file with Jupyter Notebook or another IDE（Vs code or Pycharm).  
- You will also need to install the OpenCV (CV2) library to run the code.  

### Pre-trained Model  

- The file `model_keras_2.h5` is the finished model that can be used for predictions.  
- This model was trained using Kaggle. You can find the notebook [here](https://www.kaggle.com/code/moonquakemiao/deep-cnn-for-fer-git-on-comments).  

## How to run

Download cv2 and keras and numpy
```
# Install process only for reference, there maybe some version problem when installing
pip install opencv-python
pip install keras
pip install numpy
```

Change path to the model in .ipynb or .py

Or you can use model_keras.h5 directly

```
model = load_model('PATH_TO_THE_MODEL')
model = load_model('model_keras.h5')
```

Run python

```
python emotion_recognition.py
```

## Note  

Please be aware that the case study topics may vary each year. Wishing all students a successful graduation in their senior year!（edit on Nov 2024)

## Update version (edit on Dec 2024)
Due to some important reason, we add the new version about this Model demo, in other branch. Here is the tips:
- 1111.mp4 is a test video
- emotion_result.csv and json is two version of the output
- the running demo is the 'Untitled.ipynb', there are three block:
  - the origin block which can run with video
  - the second block is which can run with video and output is csv
  - the third block is which can run with video and output is json, which is better to add them into RAG


这个存储库用于com3025j ARVR案例研究代码，它可以在jupyter notebook的ptyhon环境中运行。

在此文件中，demo的格式为ipynb文件，用户需要使用jupyter notebook或其他IDE打开该文件，并且还需要安装CV2库才能运行代码。

对于model_keras.h5，这个文件是完成的模型，可以用于预测。

这个模型是由Kaggle训练的模型，环境是kaggle自带的，具体环境不清楚


