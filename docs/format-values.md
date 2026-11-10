# Opt-in String/Character FORMAT lists

The per-Variant getter `variant.format_values("VS")` reads the native padded
FORMAT string bytes directly into per-sample Python lists, without constructing
a NumPy string array or a full intermediate sample string.
`variant.format("VS")` is unchanged.

For sample strings `a%2Cb,c`, `.`, and `x,,y,`, the new getter returns
`[["a%2Cb", "c"], ["."], ["x", "", "y", ""]]`. It splits literal commas before
UTF-8 decoding, stops at native NUL padding, and preserves percent escapes,
literal missing dots, empty tokens and sample order. These are string lists,
not inferred numeric values; no missing/fill policy is imposed.

An absent field returns `None`, distinct from a present sample containing `"."`.
With zero selected samples, the result is `[]`. Unknown header fields raise
`KeyError`; numeric FORMAT fields and raw GT raise `TypeError`. Invalid UTF-8
raises `UnicodeDecodeError`, not silent replacement. Returned lists/strings own
their data and remain usable after reader advance/close and Variant destruction.
