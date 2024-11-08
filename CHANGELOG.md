# MolGraph 0.7.8

## Version 0.7.8 (2024-11-08)

## Minor features and improvements
- `molgraph.layers`
    - Added layer `UpdateField`.


## Version 0.7.7 (2024-10-31)

## Bug fixes
- `molgraph.applications.proteomics`
    - Fix some bugs, and update default config.


## Version 0.7.6 (2024-10-30)

## Minor features and improvements
- `molgraph.applications.proteomics`
    - Two different types of peptide models now exist --- one with, and one without, virtual/super nodes. For inclusion of super nodes specify `super_nodes=True` for `PeptideGraphEncoder`, otherwise `False`. Depending on `super_nodes` parameter, `PeptideModel` (aliased `PeptideGNN` or `PepGNN`) will return a Keras Sequential model with an certain readout layer. 
- `molgraph.models.interpretability`
    - Add `reduce_features` argument (default True) to `GradientActivationMapping`. Specifies whether node feature dimension should be averaged.


## Version 0.7.5 (2024-09-26)

## Major features and improvements
- `molgraph.applications.proteomics`
    - `PeptideSaliency` is now based on the gradient (class) activation mapping algorithm. It considers all node features, including intermediate ones. Based on preliminary experiments, the saliencies looks more reasonable now.
    - Users are now able to add their own dictionary of AA/Residue SMILES, see README at `molgraph/applications/proteomics/`.


## Version 0.7.4 (2024-09-20)

## Bug fixes
- `molgraph.applications.proteomics`
    - Remove `keras.Sequential` wrapping around RNN and DNN of `PeptideGNN` to avoid 'graph disconnect' error.


## Version 0.7.3 (2024-09-18)

## Minor features and improvements
- `chemistry.MolecularGraphEncoder` are now by default *not* computnig positional encoding. Pass an `integer` to `positional_encoding_dim` to compute positional encodings of dim `integer`.


## Version 0.7.2 (2024-09-06)

## Bug fixes
- add `MANIFEST.in` and modify `setup.py` to include json files.


## Version 0.7.1 (2024-09-06)

## Bug fixes
- `molgraph.applications.proteomics`
    - Fix import issue.


## Version 0.7.0 (2024-09-05)

## Major features and improvements
- `molgraph.applications`
    - Adding an application (`proteomics`). Applications are somewhat experimental to begin with and thus potentially subject to changes. See application README for updates.


## Version 0.6.16 (2024-09-04)

### Minor features and improvements

- `molgraph.models.interpretability`
    - `Saliency` takes a new argument, `absolute` (True/False), which decides whether the gradients should be absolute or not. Namely, if `absolute=False` (which is default), saliency values will be both negative and positive.
    

## Version 0.6.15 (2024-09-03)

### Major features and improvements

- `molgraph.layers`
    - `SuperNodeReadout` added. This layer extracts "super node" features based on an indicator field. Basically, it performs a boolean_mask on node features resulting in a `tf.Tensor` of shape (n_subgraphs, n_supernodes, n_features). This tensor can then be inputted to a sequence model such as an RNN.

## Version 0.6.14 (2024-09-02)

### Bug fixes
- `molgraph.models.interpretability` and `molgraph.layers.gnn`
    - `GradientActivationMapping` now behaves as expected, when using `GNN`. A private method was implemented that "watches" the intermediate inputs.


## Version 0.6.13 (2024-09-02)

### Major features and improvements
- `molgraph.layers`
    - The default kernel initializer is now again 'glorot_uniform'. This is the default kernel initializer for `keras.layers.Dense` and seems to work well for the GNN layers as well. To use set the previous default kernel initializer, specify `kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.005)`.


## Version 0.6.12 (2024-09-02)

### Major features and improvements

- `molgraph.layers`
    - A GNN layer ("`GNN`") is implemented to combine the output of the last GNN layer as well as all the intermediate GNN layers. Simply pass a list of GNN layers to `GNN`: (`GNN([..., GINConv(128), GINConv(128), ...])`) and pass it
    as a layer to e.g. `keras.Sequential`. 


## Version 0.6.11 (2024-08-27)

### Minor features and improvements

- `molgraph.models.interpretability`
    - `GradientActivationMapping` deprecates `layer_names` and will be default watch the node features of all graph tensors. 

## Version 0.6.10 (2024-08-23)

### Bug fixes
- `molgraph.tensors`
    - `GraphTensor` can now add field with prepended underscore in the input pipeline (after batching etc.) 

## Version 0.6.9 (2024-07-08)

### Bug fixes

- `molgraph.models`
    - `molgraph.models.interpretability` models now work with multi-label data, as it was supposed to.
    - `molgraph.models.interpretability.GradientActivationMapping` now computes alpha correctly, namely, computed for each subgraph separately (based on the graph indicator).

### Minor features and improvements

- `molgraph.models`
    - `molgraph.models.interpretability` models are now simplified and are not by default wrapped in tf.function; if desirable, the user may wrap it in tf.function themselves.


## Version 0.6.8 (2024-07-04)

### Bug fixes

- `molgraph.layers`
    - `molgraph.layers.GNNLayer`'s get_config and from_config methods are updated to allow for serialization of GNN models. 
    - `molgraph.layers.GNNInputLayer` and `molgraph.layers.GNNInput` were added to allow for serialization of GNN models.
    - `molgraph.layers.StandardScaling`, `molgraph.layers.Threshold` and `molgraph.layers.CenterScaling` can now be loaded.
- `molgraph.chemistry`
    - `molgraph.chemistry.Tokenizer` now appropriately adds self loops (if specified).

## Version 0.6.7 (2024-05-08)

### Fixes
- `molgraph.layers.GATv2Conv` should now better correspond to the original GATv2 implementation. 

## Version 0.6.6 (2024-03-12)

### Bug fixes
- MolGraph should now install appropriate tensorflow version.

## Version 0.6.5 (2024-01-03)

MolGraph can now be installed (via pip) for GPU and CPU users: `pip install molgraph[gpu]` and `pip install molgraph`, respectively.

## Version 0.6.4 (2023-11-30)

### Major features and improvements
- `molgraph.models`
    - `molgraph.models.gin` now considers initial node features (which has been subject to a linear transformation) in its output.


## Version 0.6.3 (2023-11-27)

### Breaking changes
- `molgraph.tensors`
    - `molgraph.tensors.graph_tensor` can no longer be stacked. To stack GraphTensor instances, perform `tf.concat` followed by `.separate()`.


## Version 0.6.2 (2023-11-27)

### Bux fixes
- `molgraph.tensors`
    - `molgraph.tensors.graph_tensor` now accepts list of values, with sizes set to None.

## Version 0.6.1 (2023-11-27)

### Breaking changes
- `molgraph.tensors` 
    - `molgraph.tensors.graph_tensor` deprecates old features, attributes, etc. See documentation for how to use the GraphTensor.

### Bug fixes
- `molgraph.chemistry`
    - `molgraph.chemistry.encoders` now compatible with latest RDKit version.


## Version 0.6.0 (2023-08-31)

### Major features and improvements
- `molgraph.tensors`
    - `GraphTensor` is now implemented with the `tf.experimental.ExtensionType` API. **Be aware that this migration will likely break user code**. The migration was deemed necessary to make the MolGraph API more robust, reliable and maintainable. The `GraphTensor` is by default in its non-ragged (disjoint) state when obtained from the `MolecularGraphEncoder`. A non-ragged `GraphTensor` is now, thanks to the `tf.experimental.ExtensionType` API batchable. There is no need to `.separate()` it before using it with `tf.data.Dataset`. Furthermore, no need to add type_spec to keras.Sequential model. 
- `molgraph.layers`
    - `molgraph.layers.GINConv` now optionally updates edge features at each layer, given that `use_edge_features=True` and `edge_features` exist. Specify `update_edge_features=True` to update edge features at each layer. If `update_edge_features=False`, `GINConv` will behave as before, namely, edge_features will only be updated if `edge_dim!=node_dim`. Furthermore, `GINConv` uses edge features by default, given that edge features exist (specify `use_edge_features=False` to not use edge features).
- `molgraph.models`
    - Note on `models`: models are currently being experimented with and will likely change in the future.
    - `GIN` model implemented, based on the original paper. This model differ somewhat from a GIN implementation using `keras.Sequential` with `GINConv` layers, as it outputs node embeddings from each `GINConv` layer. These embeddings can be used for graph predictions, by performing readout on each embedding and conatenating it. Or it can be used for node/edge predictions by simply concatenating these embeddings. The embeddings outputted from the GIN model is stored in `node_feature` as a 3-D tf.Tensor (or 4-D tf.RaggedTensor). 
    - `DMPNN` model changed to better match the implementation of the original paper (though some differences may still exist).

### Minor features and improvements
- `molgraph.layers`
    - `normalization` is now by default set to None for most gnn layers (which means no normalization should be applied). 
- `molgraph.chemistry`
    - `tf_records` module now implements `writer`: a context manager which makes it easier to write tf records to file. `writer` seems to work fine, though it might be subject to changes in the future. Note: `tf_records.write` can still be used, though the `device` argument is depracated. To write tf records on CPU (given that GPU is available and used by default), use `with tf_records.writer(path) as writer: writer.write(data={'x': ...}, encoder=...)` instead.

### Breaking changes
- `molgraph.models`
    - `DGIN` model has been removed. 
    - `DMPNN` now takes different parameters.
        - The first parameter of `DMPNN` and `MPNN` is `steps` and not `units` (`units` is the second parameter). 

## Version 0.5.8 (2023-08-11)

### Bug fixes
- `molgraph.layers`
    - `from_config` of `molgraph.layers.gnn_layer` should now properly build/initialize the the derived layer. Specifically, a `GraphTensorSpec` should now be passed to `build_from_signature()`. 

### Minor features and improvements
- `molgraph.models`
    - `layer_names` of `molgraph.models.GradientActivationMapping` is now optional. If `None` (the default), the object will look for, and use, all layers subclassed from `GNNLayer`. If not found, an error will be raised.
 

## Version 0.5.7 (2023-07-20)

### Breaking changes
- `molgraph`
    - Optional `positional_encoding` field of `GraphTensor` is renamed to `node_position`. A (Laplacian) positional encoding is included in a `GraphTensor` instance when e.g. `positional_encoding_dim` argument of `chemistry.MolecularGraphEncoder` is not `None`. The positional encoding is still referred to as "positional" and "encoding" in `layers.LaplacianPositionalEncoding` and `chemistry.MolecularGraphEncoder`, though the actual data field added to the `GraphTensor` is `node_position`.  
- `molgraph.chemistry`
    - `inputs` argument replaced with `data`.

### Bug fixes 
- `molgraph.chemistry`
    - `molgraph.chemistry.tf_records.write()` no longer leaks memory. A large dataset (about 10 million small molecules, encoded as graph tensors) is expected to be written to tf records without exceeding 3GB memory usage. 

### Minor features and improvements
- `molgraph.chemistry`
    - `molgraph.chemistry.tf_records.write()` now accepts `None` input for `encoder`. If `None` is passed, it is assumed that `data['x']` contains `GraphTensor` instances (and not e.g. SMILES strings).
- `molgraph.tensors`
    - `node_position` is now an attribute of the `GraphTensor`. Note: `positional_encoding` can still be used to access the positional encoding (now `node_position` of a `GraphTensor` instance). However, it will be depracated in the near future.


## Version 0.5.6 (2023-07-19)

### Breaking changes
- `molgraph.layers`
    - `molgraph.layers.DotProductIncident` no longer takes `apply_sigmoid` as an argument. Instead it takes `normalize`, which specifies whether the dot product should be normalized, resulting in cosine similarities (values between -1 and 1).
- `molgraph.models`
    - `GraphAutoEncoder` (GAE) and `GraphVariationalAutoEncoder` (GVAE) are changed. The default `loss` is `None`, which means that a default loss function is used. This loss function simply tries to maximize the positive edge scores and minimize the negative edge scores. `predict` now returns the (positive) edge scores corresponding to the inputted `GraphTensor` instance. `get_config` now returns a dictionary, as expected. The default decoder is `molgraph.layers.DotProductIncident(normalize=True)`. Note: there is still some more work to be done with GAE/GVAE; e.g. improving the "`NegativeGraphSampler`" and (for VGAE) improving the `beta` schedule.
- `molgraph.tensors`
    - `GraphTensor.propagate()` now removes the `edge_weight` data component, as
    it has already been used.


### Major features and improvements
- `molgraph.models`
    - `GraphMasking` (alias: `MaskedGraphModeling`) is now implemented. Like the autoencoders, this model pretrains an encoder; though instead of predicting links between nodes, it predicts randomly masked node and edge features. (Currently only works with tokenized node and edge features (via `chemistry.Tokenizer`).) This pretraining strategy is inspired by BERT for language modeling.

### Bug fixes 
- `molgraph.layers`
    - `from_config` now works as expected for all gnn layers. Consequently, `gnn_model.from_config(gnn_model.get_config())` now works fine.

### Minor features and improvements
- `molgraph.layers`
    - `_build_from_vocabulary_size()` removed from `EmbeddingLookup`. Instead creates `self.embedding` in `adapt()` or `build()`.
    

## Version 0.5.3-0.5.5 (2023-07-17)

### Bug fixes
- `molgraph`
    - Make molgraph compatible with tf>=2.9.0. Before only compatible with tf>=2.12.0.
- `molgraph.layers`
    - `_get_reverse_edge_features()` of `edge_conv.py` is now correctly obtaining the reverse edge features.
    - Missing numpy import is now added for some preprocessing layers.


## Version 0.5.2 (2023-07-11)

### Breaking changes

- `molgraph.models`
    - Update DGIN and DMPNN. These models are now working more as expected.


## Version 0.5.1 (2023-07-10)

- `molgraph`
    - Replace tensorflow/keras functions to make MolGraph compatible with tensorflow 2.13.0. E.g. `keras.utils.register_keras_serializable` is replaced with `tf.keras.saving.register_keras_serializable`.


## Version 0.5.0 (2023-07-07)

### Breaking changes

- `molgraph.layers`
    - New base layer for GNN layers: `molgraph.layers.GNNLayer`. Old base layer `molgraph.layers.BaseLayer` is removed. New base layer can be used to define new GNN layers. In brief these are the changes that may affect the user: (1) `subclass_call` and `subclass_build` are renamed to `_call` and `_build`; (2) `_build` is replaced by `build_from_signature` which accepts a `GraphTensor` or `GraphTensorSpec` instead of tensors or tensor specs; which means that when building layers, you can obtain shapes from all nested data components corresponding to the `GraphTensor` input.
    - Base layer file changed from `molgraph/layers/base.py` to `molgraph/layers/gnn_layer.py`
    - Layer ops file changed from `molgraph/layers/ops.py` to `molgraph/layers/gnn_ops.py`
    - `batch_norm` replaced with `normalization` for the built-in GNN layers as well as the base gnn layer. And if set to True, `keras.layers.LayerNormalization` will be used. Specify `normalization='batch_norm'` to use `keras.layers.BatchNormalization`.
    - The attribute `edge_feature` (as well as `node_feature`, `edge_src`, `edge_dst` and `graph_indicator`) always exist in a graph tensor instance. If you need to check whether e.g. edge features exist in the graph, check whether the attribute `edge_feature` is not None (`graph_tensor.edge_feature is not None`), instead of checking whether the attribute `edge_feature` exist (`hasattr(graph_tensor, 'edge_feature'`), which will always be True.
- `molgraph.models`
    - Saliency and gradient activation models are now implemented as `tf.Module`s instead of `keras.Model`s. They no longer have a `predict()` method and should instead be called directly via `__call__()`. If batching is desired, loop over a `tf.data.Dataset` manually and withing the loop pass the graph tensor instance (and optionally the label) as `__call__(x, y)`.
- `molgraph.chemistry`
    - Removed deprecated chemistry featurizers and tokenizers (e.g. `chemistry.AtomicFeaturizer` and `chemistry.AtomicTokenizer`, etc.). Simply use `chemistry.Featurizer` or `chemistry.Tokenizer` instead (for both atoms and bonds).

### Major features and improvements

- `molgraph.layers`
    - Allows derived GNN layers (inherting from `GNNLayer`) to optionally pass `update_step` to override the default update step (`_DefaultUpdateStep` in `gnn_layer.py`). The custom `update_step` should be a `keras.layers.Layer` which takes as input both the updated node (or edge) features ("inputs") as well as the previous node (or edge) features ("states"/residuals). One example of GNN layer which supplies a custom `update_step` ( `_FeedForwardNetwork`) is the `molgraph.layers.GTConv`.
- `molgraph.tensors`
    - Added `propagate()` method to `GraphTensor` which propagates node features within the graph. Most built-in GNN layers now utilizes this method to propagate node features.
- `tests`
    - Adding more extensive/systematic unit tests.

### Minor features and improvements

- `molgraph.tensors`
    - `update()` method of `GraphTensor` now accepts keyword arguments. E.g. `graph_tensor.update(node_feature=node_feature_updated)` is valid.
    - Added `is_ragged()` method to `GraphTensor` which checks whether nested data is ragged. I.e., whether the graph tensor instance is in a ragged state.
    - Added several properties to `GraphTensor` and `GraphTensorSpec`: `node_feature`, `edge_src`, `edge_dst`, `edge_feature`, `graph_indicator`. Previously these properties were accessed via `__getattr__`, now they are not. Conveniently, if they do not exist (e.g. `edge_feature` is non-existent in the graph tensor instance) None is returned. Note: new data components added by the user (e.g. `node_feature_updated`) can still be accessed as attributes.
- Misc
    - Cleaned up code (mainly for the GNN layers) and modified/improved docstrings.

### Bug fixes

- `molgraph.chemistry`
    - `features.GasteigerCharge()` (and possibly other features) no longer gives None, nan or inf values. 
- `molgraph.models`
    - Saliency and gradient activation mappings now works with `tf.saved_model` API.
    - Saliency and gradient activation mappings now work well with both ragged and non-ragged GraphTensor, as well as an optional label (for multi-label and multi-class classification). Note that these modules automatically sets an `input_signature` for `__call__` upon first call. 


## Version <0.5.0 (202X-XX-XX)
### \[...\]



