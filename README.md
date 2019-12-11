# TPGST reimplementation with pytorch
### Paper: [PREDICTING EXPRESSIVE SPEAKING STYLE FROM TEXT IN END-TO-END SPEECH SYNTHESIS](https://arxiv.org/pdf/1808.01410.pdf)

## Prerequisite
- python 3.7
- pytorch 1.3
- librosa, scipy, tqdm, tensorboardX

## Dataset
- [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset), Korean female single speaker speech dataset.

## Samples
- [Post](https://yangyangii.github.io/2019/12/11/TPGST.html)

## Usage
1. Download the above dataset and modify the path in config.py. And then run the below command.
    ```
    python prepro.py
    ```

2. The model needs to train 100k+ steps
    ```
    python train.py <gpu_id>
    ```

3. After training, you can synthesize some speech from text.
    ```
    python synthesize.py <gpu_id> <model_path>
    ```

4. To listen your samples, you may need mel2wav vocoder. I didn't include vocoder in this repo.


## Notes
- I think it is not different much on KSS dataset.
- I will be doing more experiminets soon.
