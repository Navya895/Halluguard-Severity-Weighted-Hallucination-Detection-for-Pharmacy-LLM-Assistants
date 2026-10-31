import torch

cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

if cuda_available:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    print(f"PyTorch CUDA version: {torch.version.cuda}")
