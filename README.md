# 🌸 Federated Learning with Flower & Hydra

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flower](https://img.shields.io/badge/Flower-FL-green.svg)](https://flower.dev)
[![Hydra](https://img.shields.io/badge/Hydra-Config-orange.svg)](https://hydra.cc)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org)

> 🚀 A production-ready federated learning framework combining **Flower** for distributed ML and **Hydra** for configuration management. Train neural networks across multiple clients while keeping data decentralized!

## 🎯 Features

- 🏗️ **Modular Architecture**: Clean separation of concerns with dedicated modules
- ⚙️ **Flexible Configuration**: Hydra-powered YAML configurations for easy experimentation  
- 🔄 **Federated Learning**: Distributed training using Flower framework
- 🧠 **Neural Networks**: CNN implementation for MNIST classification
- 📊 **Multiple Clients**: Simulate federated scenarios with configurable client numbers
- 💾 **Result Persistence**: Automatic saving of training history and metrics

## 🏛️ Architecture

📋 ---- Step 1: Configuration

⚙️ base.yaml + toy.yaml → 🎛️ Hydra loads configs

🎮 ---- Step 2: Orchestration

📄 main.py → Coordinates everything
📊 dataset.py → Splits MNIST data into client chunks
👥 client.py → Creates federated learning clients
🖥️ server.py → Sets up aggregation strategy

🌸 ---- Step 3: Federated Training

🎯 Flower server distributes global model to clients
👤 Each client trains on their local data partition
🧠 model.py → CNN trains on MNIST digits (0-9)
📤 Clients send updated parameters back to server
🔄 Server aggregates all client updates
📈 Process repeats for multiple rounds

💾 ---- Step 4: Results

📦 Training history saved to results.pkl
⚙️ Configuration snapshot saved automatically
## 📁 Project Structure

```
📦 federated-learning-project/
├── 📄 main.py              # Main orchestration script
├── 🎮 toy.py               # Hydra configuration examples
├── 📊 dataset.py           # MNIST data loading & partitioning
├── 👥 client.py            # Flower client implementation
├── 🖥️ server.py            # Server-side functions
├── 🧠 model.py             # Neural network definition & training
├── 📁 conf/                # Configuration directory
│   ├── ⚙️ base.yaml        # Main configuration
│   └── 🎯 toy.yaml         # Example configurations
└── 📁 outputs/             # Generated results (auto-created)
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision flwr hydra-core omegaconf
```

### 🎮 Interactive Demo

<details>
<summary>🎯 Try the Toy Example First!</summary>

```bash
python toy.py
```

This demonstrates Hydra's configuration magic:
- ✨ YAML config parsing
- 🔧 Function instantiation  
- 🏗️ Object creation from config
- 🎪 Partial function application

</details>

### 🏃‍♂️ Run Federated Learning

```bash
# Use default configuration
python main.py

# Override specific parameters
python main.py num_clients=5 num_rounds=10

# Use different config file
python main.py --config-name=custom_config
```

## ⚙️ Configuration

### 📋 Key Parameters in `base.yaml`

```yaml
# Client Configuration
num_clients: 10                    # Total federated clients
num_clients_per_round_fit: 3       # Clients per training round
num_clients_per_round_eval: 3      # Clients per evaluation round

# Training Parameters  
num_rounds: 3                      # Federated learning rounds
batch_size: 32                     # Local batch size
num_classes: 10                    # MNIST classes (0-9)

# Strategy Configuration
strategy:
  _target_: flwr.server.strategy.FedAvg
  fraction_fit: 0.3
  min_fit_clients: 2
  min_available_clients: 10
```

### 🎛️ Customization Examples

<details>
<summary>💡 Click to see configuration examples</summary>

**Change the model:**
```yaml
model:
  _target_: model.Net
  num_classes: 10
```

**Adjust training settings:**
```yaml
config_fit:
  lr: 0.01
  momentum: 0.9  
  local_epochs: 5
```

**Modify federated strategy:**
```yaml
strategy:
  _target_: flwr.server.strategy.FedProx
  proximal_mu: 0.1
```

</details>

## 🧠 Model Architecture

The CNN model (`Net` class) features:

- 🔹 **Conv Layer 1**: 1→6 channels, 5×5 kernel
- 🔹 **Conv Layer 2**: 6→16 channels, 5×5 kernel  
- 🔹 **Max Pooling**: 2×2 after each conv layer
- 🔹 **Fully Connected**: 16×4×4 → 120 → 84 → 10 classes
- 🔹 **Activation**: ReLU throughout

## 📊 Results & Metrics

After training, find your results in:
- 📈 `outputs/{timestamp}/results.pkl` - Complete training history
- 📋 `outputs/{timestamp}/.hydra/config.yaml` - Used configuration
- 📝 `outputs/{timestamp}/.hydra/hydra.yaml` - Hydra metadata

## 🔧 Advanced Usage

### 🎯 Custom Strategies

Implement your own federated strategy:

```python
from flwr.server.strategy import Strategy

class MyCustomStrategy(Strategy):
    # Your implementation here
    pass
```

### 📊 Custom Models

Add new models to `model.py`:

```python
class MyCustomNet(nn.Module):
    def __init__(self, num_classes):
        # Your architecture here
        pass
```

### 🎮 Experiment Tracking

The framework automatically saves:
- ✅ Training/validation losses
- ✅ Accuracy metrics per client
- ✅ Model parameters after each round
- ✅ Configuration snapshots

## 🐛 Troubleshooting

<details>
<summary>❗ Common Issues & Solutions</summary>

**CUDA Memory Error:**
```bash
# Reduce batch size or use CPU
python main.py batch_size=16
```

**Configuration Not Found:**
```bash
# Check config path
python main.py --config-path=./conf --config-name=base
```

**Client Connection Issues:**
```bash
# Reduce concurrent clients
python main.py client_resources.num_cpus=1
```

</details>

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌟 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

## 📚 Learn More

- 🌸 [Flower Documentation](https://flower.dev/docs/)
- ⚡ [Hydra Documentation](https://hydra.cc/docs/intro/)
- 🔥 [PyTorch Tutorials](https://pytorch.org/tutorials/)
- 📖 [Federated Learning Paper](https://arxiv.org/abs/1602.05629)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- 🌸 **Flower Team** for the fantastic federated learning framework
- ⚡ **Facebook Research** for Hydra configuration management  
- 🔥 **PyTorch Team** for the deep learning framework
- 🎯 **MNIST Dataset** creators for the benchmark dataset

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

*Built with ❤️ for the federated learning community*

</div>
