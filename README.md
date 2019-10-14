# Dependency Parsing as Sequence Labeling with Multi-Task Learning (MTL)

This code is an upgraded version that runs on:

* ```Python 3.6```
* ```PyTorch 1.2.0```

 Dependency trees can be encoded into labels by:
 
 ```bash
python encode_dep2labels.py --file_to_encode x --output x --task x --enc x
```
 where:
 
```Python
file_to_encode=...  # file with dependencies in the CONLL format
output=...    # output with encoded trees as labels
task=... # single or multitask learning of labels [single, combined, multi] *combined only applicable for encoding 3
enc=...   # type of encoding [2,3,4]
```

#### Train a model

 ```bash
python main.py --config x 
```

* ```--train-config``` an example of a [config file for training](https://github.com/mstrise/dep2label-up/blob/master/config/train.config)
#### Parse with a pre-trained model

 ```bash
python decode.py --test x --gold x --model x/mod --status decode --gpu True --output x --ncrfpp x
```
where:

```Python
test=...  # file with encoded dependency trees as labels
gold=...    # file with depedencies in the CONLL format
model=... # path to the model directory ending with /mod
output=...   # output file with predicted dependencies in CONLL format
ncrfpp=... # path to the NCRF++ directory
```

## Acknowledgements

This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150).

## Reference

If you wish to use our work for research purposes, please cite us!
```
@inproceedings{strzyz-etal-2019-viable,
    title = "Viable Dependency Parsing as Sequence Labeling",
    author = "Strzyz, Michalina  and
      Vilares, David  and
      G{\'o}mez-Rodr{\'\i}guez, Carlos",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1077",
    doi = "10.18653/v1/N19-1077",
    pages = "717--723",
    abstract = "We recast dependency parsing as a sequence labeling problem, exploring several encodings of dependency trees as labels. While dependency parsing by means of sequence labeling had been attempted in existing work, results suggested that the technique was impractical. We show instead that with a conventional BILSTM-based model it is possible to obtain fast and accurate parsers. These parsers are conceptually simple, not needing traditional parsing algorithms or auxiliary structures. However, experiments on the PTB and a sample of UD treebanks show that they provide a good speed-accuracy tradeoff, with results competitive with more complex approaches.",
}
```

Original paper for [NCRF++](https://github.com/jiesutd/NCRFpp)

```
@inproceedings{yang2018ncrf,
 title={NCRF++: An Open-source Neural Sequence Labeling Toolkit},
 author={Yang, Jie and Zhang, Yue},
 booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
 Url = {http://aclweb.org/anthology/P18-4013},
 year={2018}
}
```

## Contact

Any questions? Bugs? Comments? Contact me using michalina.strzyz[at]udc.es
