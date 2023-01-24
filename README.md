
<h1 align="center">
  <br>
  Question Answering with NLP 💬
  <br>
</h1>

<h4 align="center">NLP Course project</h4>

<p align="center">
 <a href="#about">About the project</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a>
</p>


## About
Question answering is a key task of NLP. The most common way to approach this problem is with so called span selection. However this only allows for a single answer whereas the actual answer may be contained in multiple spans together. This project therefore looks at using a IOB-tagging system to classify the what parts of the text answer the given question if there is any answer.

## Key Features

* Experiment tracking with Weights and Biases
* Clean preprocessing pipeline
* Various experiments and models tested

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and Python3 installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/christian2903/QuestionAnswering

# Go into the repository
$ cd QuestionAnswering

# Install dependencies
$ pip3 install -r requirements.txt

# Run the experiments
$ python3 main.py
```


## License

MIT
