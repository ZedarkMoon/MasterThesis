|            | TFT | Coop | Defect |
|------------|------|--------|-----|
| Q Table    |  'CC': {'C': 15.0, 'D': 14.600000000000001}, 'CD': {'C': 15.0, 'D': 14.600000000000001}, 'DC': {'C': 12.0, 'D': 10.600000000000001}, 'DD': {'C': 12.0, 'D': 10.600000000000001}    |   'CC': {'C': 23.0, 'D': 25.0}, 'CD': {'C': 0, 'D': 0}, 'DC': {'C': 23.0, 'D': 25.0}, 'DD': {'C': 0, 'D': 0}     |   'CC': {'C': 0, 'D': 0}, 'CD': {'C': 3.9999999999999987, 'D': 4.999999999999998}, 'DC': {'C': 0, 'D': 0}, 'DD': {'C': 3.9999999999999987, 'D': 4.999999999999998} |
| Coop%      |  94.99%    |  4.96%      |  5.03%   |
| Avg Reward |  2.94741    |   4.9009     |   0.94967  |


# Cooperative Agent 
tensor([[15.0000,  5.4448]], grad_fn=<AddmmBackward>)
0 - 0: 0
tensor([[15.3858,  8.5706]], grad_fn=<AddmmBackward>)
0 - 1: 0
tensor([[ 9.4821, 25.0000]], grad_fn=<AddmmBackward>)
1 - 0: 1
tensor([[10.4479, 25.6465]], grad_fn=<AddmmBackward>)
1 - 1: 1

# Defective Agent
tensor([[0.1123, 0.2274]], grad_fn=<AddmmBackward>)
0 - 0: 1
tensor([[0.1123, 0.1404]], grad_fn=<AddmmBackward>)
0 - 1: 1
tensor([[2.2505, 4.7469]], grad_fn=<AddmmBackward>)
1 - 0: 1
tensor([[2.1010, 5.0000]], grad_fn=<AddmmBackward>)
1 - 1: 1

# Tit-For-Tat Agent
tensor([[22.5926, 24.4907]], grad_fn=<AddmmBackward>)
0 - 0: 1
tensor([[ 9.5367e-07, -9.1880e-01]], grad_fn=<AddmmBackward>)
0 - 1: 0
tensor([[23.0436, 25.0000]], grad_fn=<AddmmBackward>)
1 - 0: 1
tensor([[4.0851, 5.0000]], grad_fn=<AddmmBackward>)
1 - 1: 1