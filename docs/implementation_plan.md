# 细粒度代码任务拆解与函数签名设计

> 目标：给出可直接落地的文件级 TODO 列表、关键类/函数签名与数据流接口，确保 2 天内 MVP 可执行。

## 1. 项目结构与模块职责

```
/src
  /data
    dataset.py          # 路径序列数据集与 DataLoader
    featurizer.py       # RDKit -> 图特征
    conformer_gen.py    # 构象生成与能量计算
  /models
    gnn_encoder.py      # 分子图嵌入
    energy_head.py      # 能量回归头
    momentum_head.py    # 环境 -> 初始动量
    step_predictor.py   # 轨迹步进预测
  /train
    train_path.py       # 训练入口
    losses.py           # 损失函数
  /sim
    simulate.py         # 轨迹模拟
  /viz
    plot_energy.py      # 能量曲线可视化
    plot_3d.py          # 3D 轨迹可视化
  /utils
    config.py           # 配置管理
    logging.py          # 日志/进度
```

---

## 2. 文件级 TODO 清单 + 函数签名

### `src/data/conformer_gen.py`
**职责**：生成构象、计算能量、形成路径序列基础数据。

**TODO**
- [ ] 解析 SMILES 生成 RDKit 分子对象
- [ ] 生成多个构象（ETKDG 或 EmbedMultipleConfs）
- [ ] 使用 MMFF/UFF 估算能量
- [ ] 输出构象与能量列表

**函数签名**
```python
from typing import List, Tuple
from rdkit import Chem


def build_molecule(smiles: str) -> Chem.Mol:
    """从 SMILES 构建 RDKit 分子对象（含氢）。"""


def generate_conformers(
    mol: Chem.Mol,
    num_confs: int = 20,
    seed: int = 42,
) -> List[int]:
    """生成构象并返回 conformer ids。"""


def compute_conformer_energy(
    mol: Chem.Mol,
    conf_id: int,
    method: str = "MMFF",
) -> float:
    """计算指定构象的力场能量。"""


def build_conformer_dataset(
    smiles: str,
    num_confs: int = 20,
    method: str = "MMFF",
) -> List[Tuple[Chem.Mol, int, float]]:
    """返回 [(mol, conf_id, energy), ...]。"""
```

---

### `src/data/featurizer.py`
**职责**：将 RDKit 构象转为图特征（PyTorch Geometric Data）。

**TODO**
- [ ] 原子特征向量（元素/度/电荷/芳香性等）
- [ ] 键特征向量（键类型/共轭/环）
- [ ] 3D 坐标特征（pos）
- [ ] 构造 Data(x, edge_index, edge_attr, pos)

**函数签名**
```python
import torch
from rdkit import Chem
from torch_geometric.data import Data
from typing import Dict


def atom_features(atom: Chem.Atom) -> torch.Tensor:
    """单原子特征向量。"""


def bond_features(bond: Chem.Bond) -> torch.Tensor:
    """单键特征向量。"""


def mol_to_graph_data(
    mol: Chem.Mol,
    conf_id: int,
) -> Data:
    """RDKit 分子 + 构象 -> PyG Data。"""


def build_feature_stats(dataset: list) -> Dict[str, float]:
    """统计特征维度、节点/边分布，用于日志。"""
```

---

### `src/data/dataset.py`
**职责**：组织路径序列 + DataLoader。

**TODO**
- [ ] 将构象能量列表转换为路径序列样本
- [ ] 输出 (state_t, env, state_t+1, energy_t, energy_{t+1})
- [ ] 支持 batch 采样

**函数签名**
```python
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PathDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
    ) -> None:
        """samples: [{"data_t": Data, "data_next": Data, "env": tensor, ...}, ...]"""

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ...


def build_path_samples(
    conformer_data: List[Tuple[Chem.Mol, int, float]],
    env: Dict[str, float],
) -> List[Dict[str, Any]]:
    """将构象序列组织成训练样本。"""
```

---

### `src/models/gnn_encoder.py`
**职责**：图嵌入编码器。

**TODO**
- [ ] GNN 层堆叠（GCN/GraphSAGE）
- [ ] 池化生成分子级嵌入
- [ ] 输出 D 维向量

**函数签名**
```python
import torch
import torch.nn as nn
from torch_geometric.data import Data


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 16,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        ...

    def forward(self, data: Data) -> torch.Tensor:
        """返回分子级嵌入 z (batch_size, out_dim)。"""
        ...
```

---

### `src/models/energy_head.py`
**职责**：能量回归头。

**TODO**
- [ ] MLP(z) -> energy_hat

**函数签名**
```python
import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        ...

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """输出能量标量 (batch_size, 1)。"""
        ...
```

---

### `src/models/momentum_head.py`
**职责**：环境条件 -> 初始动量向量。

**TODO**
- [ ] env -> p0 (D 维)

**函数签名**
```python
import torch
import torch.nn as nn


class MomentumHead(nn.Module):
    def __init__(self, env_dim: int, out_dim: int) -> None:
        super().__init__()
        ...

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        """输出初始动量向量 (batch_size, out_dim)。"""
        ...
```

---

### `src/models/step_predictor.py`
**职责**：预测轨迹步进方向。

**TODO**
- [ ] 输入 (z_t, env) -> delta_z

**函数签名**
```python
import torch
import torch.nn as nn


class StepPredictor(nn.Module):
    def __init__(
        self,
        z_dim: int,
        env_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        ...

    def forward(self, z: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """输出 delta_z (batch_size, z_dim)。"""
        ...
```

---

### `src/train/losses.py`
**职责**：定义训练损失。

**TODO**
- [ ] 路径监督损失
- [ ] 能量回归损失（可选）
- [ ] 组合损失

**函数签名**
```python
import torch


def path_loss(z_pred_next: torch.Tensor, z_true_next: torch.Tensor) -> torch.Tensor:
    """路径回归损失。"""


def energy_loss(energy_pred: torch.Tensor, energy_true: torch.Tensor) -> torch.Tensor:
    """能量回归损失。"""


def total_loss(
    z_pred_next: torch.Tensor,
    z_true_next: torch.Tensor,
    energy_pred: torch.Tensor,
    energy_true: torch.Tensor,
    lambda_energy: float = 0.1,
) -> torch.Tensor:
    """组合损失。"""
```

---

### `src/train/train_path.py`
**职责**：训练入口与主循环。

**TODO**
- [ ] 构建数据集与 DataLoader
- [ ] 模型初始化
- [ ] 训练循环 + loss 打印
- [ ] 保存权重与日志

**函数签名**
```python
import torch
from torch.utils.data import DataLoader


def train_one_epoch(
    encoder,
    stepper,
    energy_head,
    loader: DataLoader,
    optimizer,
    device: torch.device,
) -> float:
    """单个 epoch 训练，返回平均 loss。"""


def train(
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> None:
    """训练入口。"""
```

---

### `src/sim/simulate.py`
**职责**：模拟路径生成轨迹。

**TODO**
- [ ] simulate_path(z0, env) 循环预测
- [ ] 记录 z 与 energy 序列

**函数签名**
```python
import torch
from typing import List, Tuple


def simulate_path(
    z0: torch.Tensor,
    env: torch.Tensor,
    stepper,
    energy_head,
    num_steps: int = 20,
    step_size: float = 1.0,
) -> Tuple[List[torch.Tensor], List[float]]:
    """返回 z 轨迹与能量序列。"""
```

---

### `src/viz/plot_energy.py`
**职责**：绘制能量曲线。

**TODO**
- [ ] 保存能量趋势图

**函数签名**
```python
from typing import List


def plot_energy_curve(energies: List[float], out_path: str) -> None:
    """绘制能量曲线并保存。"""
```

---

### `src/viz/plot_3d.py`
**职责**：绘制 3D 轨迹。

**TODO**
- [ ] 若维度 > 3，用 PCA 降维
- [ ] 输出 3D 轨迹散点 + 连线

**函数签名**
```python
import torch
from typing import List


def project_to_3d(z_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """PCA 降维到 3D（如需要）。"""


def plot_3d_path(z_list: List[torch.Tensor], out_path: str) -> None:
    """绘制 3D 轨迹图。"""
```

---

### `src/utils/config.py`
**职责**：集中配置（路径、超参）。

**TODO**
- [ ] 统一配置

**函数签名**
```python
from dataclasses import dataclass


@dataclass
class TrainConfig:
    smiles: str = "CCCC"
    num_confs: int = 20
    emb_dim: int = 16
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-3
```

---

### `src/utils/logging.py`
**职责**：日志与进度记录。

**TODO**
- [ ] 控制台日志
- [ ] 可选 CSV

**函数签名**
```python
import logging


def setup_logger(name: str = "demo") -> logging.Logger:
    """返回 logger。"""
```

---

## 3. 最小可运行链路（MVP 执行顺序）

1. `conformer_gen.py`：生成构象 + 能量列表
2. `featurizer.py`：构造图特征
3. `dataset.py`：生成路径样本
4. `gnn_encoder.py`：嵌入编码
5. `step_predictor.py`：预测步进
6. `train_path.py`：训练并保存模型
7. `simulate.py`：模拟路径并输出轨迹
8. `plot_energy.py` + `plot_3d.py`：可视化

---

## 4. 输出文件与演示产物

- `outputs/energy_curve.png`
- `outputs/path_3d.png`
- `outputs/logs.csv`（可选）
- `outputs/model.pt`

---

如果需要，我可以在此基础上进一步补充：
- 更贴近 RDKit/ PyG 的具体实现细节
- 训练脚本的 CLI 参数设计
- 可视化的具体风格与示例输出格式
