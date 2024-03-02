# Sharding

Sharding is a technique to partition the computation and the model across multiple accelerators.
This codebase supports flexible model and data parallelism for training and serving.

The sharding can be specified using the `mesh_dim` command line argument. The `mesh_dim` is a
comma separated list of integers representing the parallelism mesh axis dimensions. One of the
axis dimensions can be `-1`, which means that the axis dimension will be inferred based on the
total number of accelerators.

The first axis of the mesh is used for data parallelism (`dp`), the second axis used for fully sharded
data parallelism (`fsdp`), the third axis is used for tensor parallelism (`tp`), the last axis is used for
sequence parallelism (required for ring attention) (`sp`).

For example, `mesh_dim='1,64,4,1'` means 1 data parallelism, 64 fully sharded data parallelism, 4 tensor parallelism, and 1 sequence parallelism. While `mesh_dim='1,1,4,64'` means 1 data parallelism, 1 fully sharded data parallelism, 4 tensor parallelism, and 64 sequence parallelism for RingAttention.

Your total number of accelerators should be equal to the product of the mesh dimensions. For example, `mesh_dim='1,64,4,1'` requires 64 accelerators, and `mesh_dim='1,1,4,64'` requires 256 accelerators.

In general, you want to use the largest possible mesh dimension for `fsdp`. Such as `mesh_dim='1,64,1,1'` is preferred over `mesh_dim='8,8,1,1'` because the former has larger `fsdp` dimensions, which allows overlapping of computation and communication, and thus better performance.

The batch size (number of sequences per batch) should be larger than or equal to `fsdp * dp`. If you think the batch size is too large, you can allocate more accelerators to `tp` and `sp` to increase the model size and sequence length.

Using `sp` to control the sequence parallelism is required to use RingAttention. `sp=8` means sharding sequence length by 8, and `sp=1` means no sharding.
 For models that use standard attention, you can set `sp=1` and use `dp`, `fsdp`, and `tp` to control the parallelism.
