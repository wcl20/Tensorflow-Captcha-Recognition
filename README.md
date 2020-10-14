# Tensorflow Captcha Recognition

Breaking Captcha using Lenet.

## Setup
Generate Virtual environment
```bash
python3 -m venv ./venv
```
Enter environment
```bash
source venv/bin/activate
```
Install required libraries
```bash
pip install -r requirements.txt
```
Creating dataset 
```
python3 annotate.py
```
Label each image by pressing the corresponding number on the keypad.

Train model
```bash
python3 train.py -d dataset
```

Test model
```
python3 test.py -i downloads
```
The programe will show 10 decrypted captchas.
