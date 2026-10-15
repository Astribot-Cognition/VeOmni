"""Rebuilding a restored iterator must preserve the training random stream."""

import copy

import pytest
import torch
from torch.utils.data import IterDataPipe, TensorDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

from veomni.data.data_loader import DistributedDataloader


class _SequentialPipe(IterDataPipe):
    def __iter__(self):
        worker = get_worker_info()
        for index in range(40):
            if worker is None or index % worker.num_workers == worker.id:
                yield torch.tensor([2 * index, 2 * index + 1], dtype=torch.float32)


def _loader(loader_type, workers, persistent, datapipe=False):
    return loader_type(
        _SequentialPipe() if datapipe else TensorDataset(torch.arange(80, dtype=torch.float32).reshape(40, 2)),
        batch_size=2,
        num_workers=workers,
        persistent_workers=persistent,
    )


def _close(iterator, workers):
    if workers:
        iterator._shutdown_workers()


@pytest.mark.parametrize(
    "workers,persistent,phase,datapipe",
    [
        (workers, persistent, phase, False)
        for workers, persistent in [(0, False), (2, False), (2, True)]
        for phase in ["mid", "boundary", "finished"]
    ]
    + [(2, True, "finished", True)],
)
def test_restore_preserves_training_rng_and_next_losses(workers, persistent, phase, datapipe):
    torch.manual_seed(73)
    loader = _loader(DistributedDataloader, workers, persistent, datapipe)
    iterator = iter(loader)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.4))
    for _ in range(3 if phase == "mid" else 20):
        batch = next(iterator)
        batch = batch if datapipe else batch[0]
        model(batch + torch.randn(2, 2))
    if phase == "finished":
        with pytest.raises(StopIteration):
            next(iterator)
    state = copy.deepcopy(loader.state_dict())
    rng = torch.get_rng_state().clone()
    expected_iterator = iterator if phase == "mid" else iter(loader)
    expected_iterator_rng = torch.get_rng_state().clone()
    expected = []
    for _ in range(5):
        batch = next(expected_iterator)
        batch = batch if datapipe else batch[0]
        loss = model(batch + torch.randn_like(batch)).square().mean()
        expected.append((batch, loss.detach(), torch.get_rng_state().clone()))

    resumed = _loader(DistributedDataloader, workers, persistent, datapipe)
    resumed.load_state_dict(state)
    torch.set_rng_state(rng)
    if phase == "boundary":
        # The native trainer creates then clears the exhausted epoch iterator.
        iter(resumed)
    resumed_iterator = iter(resumed)
    try:
        torch.testing.assert_close(torch.get_rng_state(), expected_iterator_rng, rtol=0, atol=0)
        for batch, loss, next_rng in expected:
            actual_batch = next(resumed_iterator)
            actual_batch = actual_batch if datapipe else actual_batch[0]
            actual_loss = model(actual_batch + torch.randn_like(actual_batch)).square().mean()
            torch.testing.assert_close(actual_batch, batch, rtol=0, atol=0)
            torch.testing.assert_close(actual_loss, loss, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), next_rng, rtol=0, atol=0)
    finally:
        _close(iterator, workers)
        if expected_iterator is not iterator:
            _close(expected_iterator, workers)
        _close(resumed_iterator, workers)


@pytest.mark.parametrize("workers,persistent", [(0, False), (2, False), (2, True)])
def test_fresh_iterator_keeps_native_rng_behavior(workers, persistent):
    results = []
    for loader_type in (StatefulDataLoader, DistributedDataloader):
        torch.manual_seed(109)
        loader = _loader(loader_type, workers, persistent)
        try:
            batches = []
            for _ in range(2):
                iterator = iter(loader)
                batches.extend(batch[0] for batch in iterator)
            results.append((torch.stack(batches), torch.get_rng_state().clone()))
        finally:
            _close(iterator, workers)
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
