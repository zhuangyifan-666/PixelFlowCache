from pfc.eval.sharding import compute_shard_indices


def test_four_strided_shards_cover_each_global_index_once():
    shards = [compute_shard_indices(32, 4, index, "strided") for index in range(4)]
    flattened = [value for shard in shards for value in shard]
    assert sorted(flattened) == list(range(32))
    assert len(flattened) == len(set(flattened))
    assert shards[0] == list(range(0, 32, 4))


def test_contiguous_shards_cover_partial_tail():
    shards = [compute_shard_indices(10, 4, index, "contiguous") for index in range(4)]
    assert shards == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
