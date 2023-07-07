# MolGraph 0.4.2

## Version 0.4.2 (2023-07-07)

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

## Version <0.4.2 (202X-XX-XX)
### \[...\]



