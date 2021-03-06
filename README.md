# Word-level Seq2seq headline generation

## Setup
1. Download [trained model](https://drive.google.com/open?id=1avWWfWJxc6tt2KjYn49cWv6D_KOrTz0p) and extract it into **seq2seq/models** folder 
2. Download [test data](https://drive.google.com/open?id=1Gshy_lpTpueC7L2B93EJCLjhxOPRGCEb) and extract it into **seq2seq/data** folder 

## Run 
* Jupyter Notebook demo: **headline_demo.py**
* Count BLUE and ROUGE metrics:  run **seq2seq/predict.py**  
* Rest API: 
	* run **app.py**
	* request: ```curl -i -H "Content-Type: application/json" -X POST -d '{"article":"Sample article"}' http://localhost:5000/generate_headline```


## Set parameters and train
1. Set text params in **seq2seq/news_loader.py** 
2. Set model params in **seq2seq/seq2seq_model.py**
3. Place training data into **seq2seq/data** folder and run **train.py**


N/B: 
* Stacked seq2seq is not yet supported for beam search inference 
* Encoding issues might still arise using REST API (on Windows w/ russian inputs in particular) --> for now **predict.py** and **headline_demo.py** are more preferable.



## Datasets
* https://www.kaggle.com/yutkin/corpus-of-russian-news-articles-from-lenta
* https://github.com/RossiyaSegodnya/ria_news_dataset

## Reference:
1. https://github.com/chen0040/keras-text-summarization 
2. https://github.com/devm2024/nmt_keras
3. https://vk.com/headline_gen


This is the code for Dialogue 2019 Student session paper:  
[Shevchuk A.A., Zdorovets A.I., Text Summarization with Recurrent Neural Network for Headline Generation](http://www.dialog-21.ru/media/4680/text-summarization-with-recurrent-neural-network-for-headline-generation.pdf)



