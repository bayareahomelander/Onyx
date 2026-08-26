import torch

from onyx_cuda.model import MODEL_ID, load_model


def test_load_model_and_forward_on_cuda():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    loaded = load_model()

    assert loaded.revision == loaded.model.config._commit_hash
    assert len(loaded.revision) == 40
    assert not loaded.model.training
    assert all(parameter.device == device for parameter in loaded.model.parameters())
    assert all(parameter.dtype == torch.float16 for parameter in loaded.model.parameters())

    inputs = {
        name: tensor.to(device)
        for name, tensor in loaded.tokenizer(MODEL_ID, return_tensors="pt").items()
    }
    with torch.inference_mode():
        output = loaded.model(**inputs)
    torch.cuda.synchronize(device)

    assert output.logits.shape == (*inputs["input_ids"].shape, loaded.model.config.vocab_size)
    assert output.logits.device == device
    assert torch.cuda.max_memory_allocated(device) < torch.cuda.get_device_properties(device).total_memory

    print(f"model_revision={loaded.revision}")
    print(f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")
