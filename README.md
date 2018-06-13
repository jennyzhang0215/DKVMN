# DKVMN

Dynamic Key-Value Memory Networks for Knowledge Tracing

## Built With

* [MXNet](https://github.com/dmlc/mxnet) - The framework used
* Both Python2 and Python3 are supported

### Prerequisites
* [progress](https://pypi.python.org/pypi/progress) - Dependency package

## Model Architecture

![DKVMN Architecture](https://github.com/jennyzhang0215/DKVMN/blob/master/DKVMN_architecture.png)
![DKVMN Code](https://github.com/jennyzhang0215/DKVMN/blob/master/DKVMN_code.png)

### Data format

The first line the number of exercises a student attempted.
The second line is the exercise tag sequence.
The third line is the response sequence.

 ```
    15
    1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
 ```

### Hyperparameters

--gpus: the gpus will be used, e.g "0,1,2,3"

--max_iter: the number of iterations

--test: enable testing

--train_test: enable testing after training

--show: print progress

--init_std: weight initialization std

--init_lr: initial learning rate

--final_lr: learning rate will not decrease after hitting this threshold

--momentum: momentum rate

--maxgradnorm: maximum gradient norm

--final_fc_dim: hidden state dim for final fc layer

--n_question: the number of unique questions in the dataset

--seqlen: the allowed maximum length of a sequence

--data_dir: data directory

--data_name: data set name

--load: model file to load

--save: path to save model



### Training
 ```
 python main.py --gpus 0
 ```

### Testing
 ```
 python main.py --gpus 0 --test True
 ```

## Reference Paper

Jiani Zhang, Xingjian Shi, Irwin King, Dit-Yan Yeung. [Dynamic Key-Value Memory Networks for Knowledge Tracing](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p765.pdf).
In Proceedings of the 26th International Conference on World Wide Web, 2017: 765-774.

