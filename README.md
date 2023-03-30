# Action Detection with Mediapipe : Real-Time Detector based holistic keypoints 
<br><br><br>

![image](https://user-images.githubusercontent.com/71050591/228754928-e1ac30ca-35f9-4bd1-9304-c8b5ad6ec252.png)


## Install Rquirements

Window10, only CPU, Pycharm

<br>

```
pip install -r requirements.txt
```
 
## Train
* The 'Train Dataset' was collected directly using a web camera. I trained five behaviors, and they work well.
* I collected upper body action. But if you want to train whole body detector, it's possoble. 

```
python train.py
```


## Test
* Real-time detection is possible using web camera.
* I use just CPU enviroments, it's fine.
```
python test.py
```

<br>

## Reference

<br><br><br>
