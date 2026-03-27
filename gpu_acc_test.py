import torch
import time

def test_cal(machine):
    print(f"Using device {machine}")
    # create a random tensor
    x = torch.randn(1000,1000).to(machine)
    start = time.time()
    for _ in range(100):
        x = torch.matmul(x,x)
    print(f"time taken {time.time()-start:.2f} seconds")

def gpu_cpu_time():
    # check cuda is avaiable
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"device name is {torch.cuda.get_device_name(0)}")
        test_cal(device)
    device = torch.device("cpu")
    test_cal(device)
