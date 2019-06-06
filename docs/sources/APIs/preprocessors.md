
<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L41)</span>
## Z_RandomOverSampler

```python
galaxy_ml.preprocessors.Z_RandomOverSampler(sampling_strategy='auto', return_indices=False, random_state=None, ratio=None, negative_thres=0, positive_thres=-1)
```

----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L123)</span>
## TDMScaler

```python
galaxy_ml.preprocessors.TDMScaler(q_lower=25.0, q_upper=75.0)
```


Scale features using Training Distribution Matching (TDM) algorithm

**References**

    .. [1] Thompson JA, Tan J and Greene CS (2016) Cross-platform
           normalization of microarray and RNA-seq data for machine
           learning applications. PeerJ 4, e1621.
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L206)</span>
## GenomeOneHotEncoder

```python
galaxy_ml.preprocessors.GenomeOneHotEncoder(fasta_path=None, padding=True, seq_length=None)
```

Convert Genomic sequences to one-hot encoded 2d array

**Paramaters**

- **fasta_path**: str, default None<br>
        File path to the fasta file. There could two alternative ways to set up
        `fasta_path`. 1) through fit_params; 2) set_params().
- **padding**: bool, default is False<br>
        All sequences are expected to be in the same length, but sometimes not.
        If True, all sequences use the same length of first entry by either
        padding or truncating. If False, raise ValueError if different seuqnce
        lengths are found.
- **seq_length**: None or int<br>
        Sequence length. If None, determined by the the first entry.
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L318)</span>
## ProteinOneHotEncoder

```python
galaxy_ml.preprocessors.ProteinOneHotEncoder(fasta_path=None, padding=True, seq_length=None)
```

Convert protein sequences to one-hot encoded 2d array

**Paramaters**

- **fasta_path**: str, default None<br>
        File path to the fasta file. There could two alternative ways to set up
        `fasta_path`. 1) through fit_params; 2) set_params().
- **padding**: bool, default is False<br>
        All sequences are expected to be in the same length, but sometimes not.
        If True, all sequences use the same length of first entry by either
        padding or truncating. If False, raise ValueError if different seuqnce
        lengths are found.
- **seq_length**: None or int<br>
        Sequence length. If None, determined by the the first entry.
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L383)</span>
## ImageBatchGenerator

```python
keras.preprocessing.image.ImageBatchGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)
```

----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L393)</span>
## FastaIterator

```python
galaxy_ml.preprocessors.FastaIterator(n, batch_size=32, shuffle=True, seed=0)
```

Base class for fasta sequence iterators.

**Parameters**

- **n**: int<br>
        Total number of samples
- **batch_size**: int<br>
        Size of batch
- **shuffle**: bool<br>
        Whether to shuffle data between epoch
- **seed**: int<br>
        Random seed number for data shuffling
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L413)</span>
## FastaToArrayIterator

```python
galaxy_ml.preprocessors.FastaToArrayIterator(X, y=None, fasta_file=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, n_bases=4, seq_length=1000, base_to_index=None, unk_base=None)
```

Iterator yielding Numpy array from fasta sequences

**Parameters**

- **X**: array<br>
        Contains sequence indexes in the fasta file
- **y**: array<br>
        Target labels or values
- **fasta_file**: object<br>
        Instance of pyfaidx.Fasta
- **batch_size**: int, default=32<br>
- **shuffle**: bool, default=True<br>
        Whether to shuffle the data between epochs
- **sample_weight**: None or array<br>
        Sample weight
- **seed**: int<br>
        Random seed for data shuffling
- **n_bases**: int, default=4<br>
        4 for DNA, 20 for protein
- **seq_length**: int, default=1000<br>
        Output sequence length
- **base_to_index**: dict or None<br>
- **unk_base**: str or None<br>
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L502)</span>
## FastaDNABatchGenerator

```python
galaxy_ml.preprocessors.FastaDNABatchGenerator(fasta_path, seq_length=1000, shuffle=True, seed=None)
```

Fasta squence batch data generator, online transformation of sequences
to array.

**Parameters**

- **fasta_path**: str<br>
        File path to fasta file.
- **seq_length**: int, default=1000<br>
        Sequence length, number of bases.
- **shuffle**: bool, default=True<br>
        Whether to shuffle the data between epochs
- **seed**: int<br>
        Random seed for data shuffling
    
----

<span style="float:right;">[[source]](https://github.com/ohsu-comp-bio/Galaxy-ML/blob/master/galaxy_ml/preprocessors.py#L543)</span>
## FastaProteinBatchGenerator

```python
galaxy_ml.preprocessors.FastaProteinBatchGenerator(fasta_path, seq_length=1000, shuffle=True, seed=None)
```

Fasta squence batch data generator, online transformation of sequences
to array.

**Parameters**

- **fasta_path**: str<br>
        File path to fasta file.
- **seq_length**: int, default=1000<br>
        Sequence length, number of bases.
- **shuffle**: bool, default=True<br>
        Whether to shuffle the data between epochs
- **seed**: int<br>
        Random seed for data shuffling
    