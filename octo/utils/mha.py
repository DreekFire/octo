# Dibya Ghosh
import inspect
import logging
import warnings
from functools import partial
from typing import Any, Optional, Union

import flax.linen as nn
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array


class MultiHeadDotProductAttention(nn.MultiHeadDotProductAttention):
    """Modifies Flax's MultiHeadDotProductAttention for a better auto-regressive experience.

    How to use:
        >>> X: jax.Array with shape (batch_size, seq_len, hidden_dim)
        >>> mask: jax.Array with shape (batch_size, num_heads, seq_len, seq_len)  (Warning: must be causal!)

        # Normal way to do forward prediction all at once
        >>> out1 = transformer.apply({'params': params}, X, mask)

        # Autoregressive way to do forward prediction one step at a time
        >>> cache_vars = {}
        >>> outs = []
        >>> for i in range(seq_len):
                out, cache_vars = transformer.apply({'params': params, **cache_vars}, X[:, i:i+1], mask[:, :, i:i+1], mutable=['cache'])
                outs.append(out)
        >>> out2 = jnp.concatenate(outs, axis=1)

        >>> assert jnp.allclose(out1, out2)
    """

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Optional[Array] = None,
        inputs_v: Optional[Array] = None,
        *,
        inputs_kv: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
        dropout_rng: Optional[Any] = None,
        sow_weights: bool = False,
        sampling: bool = False,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
          inputs_q: input queries of shape ``[batch_sizes..., length, features]``.
          inputs_k: key of shape ``[batch_sizes..., length, features]``. If None,
            inputs_k will copy the value of inputs_q.
          inputs_v: values of shape ``[batch_sizes..., length, features]``. If None,
            inputs_v will copy the value of inputs_k.
          inputs_kv: key/values of shape ``[batch_sizes..., length, features]``. If
            None, inputs_kv will copy the value of inputs_q. This arg will be
            deprecated soon. Use inputs_k and inputs_v instead.
          mask: attention mask of shape ``[batch_sizes..., num_heads, query_length,
            key/value_length]``. Attention weights are masked out if their
            corresponding mask value is ``False``.
          deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic.
          dropout_rng: optional rng key to pass to the attention layer's dropout
            mask. Otherwise, self.make_rng('dropout') is used instead.
          sow_weights: if ``True``, the attention weights are sowed into the
            'intermediates' collection. Remember to mark 'intermediates' as
            mutable via ``mutable=['intermediates']`` in order to have that
            collection returned.

        Returns:
          output of shape ``[batch_sizes..., length, features]``.
        """
        if inputs_kv is not None:
            if inputs_k is not None or inputs_v is not None:
                raise ValueError(
                    "If either `inputs_k` or `inputs_v` is not None, "
                    "`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` "
                    "and `inputs_v` must be None. We recommend using `inputs_k` and "
                    "`inputs_v` args, since `inputs_kv` will be deprecated soon. See "
                    "https://github.com/google/flax/discussions/3389 for more "
                    "information."
                )
            inputs_k = inputs_v = inputs_kv
            warnings.warn(
                "The inputs_kv arg will be deprecated soon. "
                "Use inputs_k and inputs_v instead. See "
                "https://github.com/google/flax/discussions/3389 "
                "for more information.",
                DeprecationWarning,
            )
        else:
            if inputs_k is None:
                if inputs_v is not None:
                    raise ValueError(
                        "`inputs_k` cannot be None if `inputs_v` is not None. "
                        "To have both `inputs_k` and `inputs_v` be the same value, pass in the "
                        "value to `inputs_k` and leave `inputs_v` as None."
                    )
                inputs_k = inputs_q
            if inputs_v is None:
                inputs_v = inputs_k
            elif inputs_v.shape[-1] == inputs_v.shape[-2]:
                warnings.warn(
                    f"You are passing an array of shape {inputs_v.shape} "
                    "to the `inputs_v` arg, when you may have intended "
                    "to pass it to the `mask` arg. As of Flax version "
                    "0.7.4, the function signature of "
                    "MultiHeadDotProductAttention's `__call__` method "
                    "has changed to `__call__(inputs_q, inputs_k=None, "
                    "inputs_v=None, *, inputs_kv=None, mask=None, "
                    "deterministic=None)`. Use the kwarg `mask` instead. "
                    "See https://github.com/google/flax/discussions/3389 "
                    "and read the docstring for more information.",
                    DeprecationWarning,
                )

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            f"Memory dimension ({qkv_features}) must be divisible by number of"
            f" heads ({self.num_heads})."
        )
        head_dim = qkv_features // self.num_heads

        dense = partial(
            nn.DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
            dot_general_cls=self.qkv_dot_general_cls,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            dense(name="query")(inputs_q),
            dense(name="key")(inputs_k),
            dense(name="value")(inputs_v),
        )

        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = nn.LayerNorm(name="query_ln", use_bias=False)(query)  # type: ignore[call-arg]
            key = nn.LayerNorm(name="key_ln", use_bias=False)(key)  # type: ignore[call-arg]

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if sampling:
            assert self.is_mutable_collection("cache")
            # detect if we're initializing by absence of existing cache data.

            max_length = mask.shape[-1] if mask is not None else inputs_k.shape[-2]

            if not self.has_variable("cache", "cached_key"):
                logging.debug("Initializing cache with max length", max_length)

            key_shape = (*key.shape[:-3], max_length, *key.shape[-2:])
            value_shape = (*value.shape[:-3], max_length, *value.shape[-2:])

            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, key_shape, key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, value_shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )

            (
                *batch_dims,
                max_length,
                num_heads,
                depth_per_head,
            ) = cached_key.value.shape
            # shape check of cached keys against query input
            expected_shape = tuple(batch_dims) + (None, num_heads, depth_per_head)
            if not all(
                [a == b or b is None for a, b in zip(query.shape, expected_shape)]
            ):
                raise ValueError(
                    "Autoregressive cache shape error, "
                    "expected query shape %s instead got %s."
                    % (expected_shape, query.shape)
                )
            n_current = query.shape[-3]
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            print("Running at index", cur_index, "to index", cur_index + n_current)
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices: tuple[Union[int, Array], ...] = (zero,) * len(batch_dims) + (
                cur_index,
                zero,
                zero,
            )
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            cache_index.value = cache_index.value + n_current

            # causal mask for cached decoder self-attention:
            # our single query position should only attend to those key
            # positions that have already been generated and cached,
            # not the remaining zero elements.
            mask = nn.combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length) < cur_index + n_current,
                    tuple(batch_dims) + (1, n_current, max_length),
                ),
            )

        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            m_deterministic = nn.merge_param(
                "deterministic", self.deterministic, deterministic
            )
            if not m_deterministic and dropout_rng is None:
                dropout_rng = self.make_rng("dropout")
        else:
            m_deterministic = True

        # `qk_attn_weights_einsum` and `attn_weights_value_einsum` are optional
        # arguments that can be used to override the default `jnp.einsum`. They
        # exist for quantized einsum support in AQT.
        qk_attn_weights_einsum = (
            self.qk_attn_weights_einsum_cls()
            if self.qk_attn_weights_einsum_cls
            else None
        )
        attn_weights_value_einsum = (
            self.attn_weights_value_einsum_cls()
            if self.attn_weights_value_einsum_cls
            else None
        )
        # apply attention
        attn_args = (query, key, value)
        # This kwargs list match the default nn.dot_product_attention.
        # For custom `attention_fn`s, invalid kwargs will be filtered.
        attn_kwargs = dict(
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            qk_attn_weights_einsum=qk_attn_weights_einsum,
            attn_weights_value_einsum=attn_weights_value_einsum,
        )
        attn_kwargs = {
            k: v
            for k, v in attn_kwargs.items()
            if k in inspect.signature(self.attention_fn).parameters
        }
        if sow_weights:
            x = self.attention_fn(*attn_args, **attn_kwargs, module=self)
        else:
            x = self.attention_fn(*attn_args, **attn_kwargs)
        # back to the original inputs dimensions
        out = nn.DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.out_kernel_init or self.kernel_init,
            bias_init=self.out_bias_init or self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            dot_general_cls=self.out_dot_general_cls,
            name="out",  # type: ignore[call-arg]
        )(x)
        return out
