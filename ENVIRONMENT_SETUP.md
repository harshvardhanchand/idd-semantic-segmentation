# Environment Setup - IDD Semantic Segmentation

## Development Strategy: Hybrid Local + Cloud Approach

**Architecture**: Local Development (Cursor) + Cloud GPU Training (Google Colab)

### Why This Approach?
- ✅ **Best IDE Experience**: Full Cursor capabilities locally
- ✅ **Free GPU Training**: Google Colab T4/A100 GPUs at no cost
- ✅ **Rapid Iteration**: Code locally, train in cloud
- ✅ **No Complex Setup**: Avoid RunPod SSH/billing complications
- ✅ **Cost-Effective**: $0 total cost vs RunPod's $0.26/hour

### Why Not RunPod?
Initial RunPod exploration revealed several friction points:
- **Setup Complexity**: SSH authentication issues, container restarts, port mapping confusion
- **Billing Anxiety**: Continuous $0.26/hour charges during development/debugging/uploads
- **File Transfer Bottlenecks**: Slow SCP uploads (2GB dataset = 30+ min + connection maintenance)
- **Development Friction**: Web terminal vs full IDE, package reinstalls after restarts

For a **3-5 hour semantic segmentation project**, RunPod was over-engineered. The hybrid approach provides **better developer experience** with **zero cost** for this scope.

## Local Development Environment

### Python Version
- **Target**: Python 3.9.18 (specified in `.python-version`)
- **Compatibility**: Python >= 3.8, < 3.12

### Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Local Capabilities
- **Code Development**: Complete pipeline in Cursor
- **Data Exploration**: Jupyter notebooks for analysis  
- **Unit Testing**: Small batch testing on CPU
- **Debugging**: Full debugging capabilities
- **Version Control**: Git integration

## Cloud Training Environment

### Google Colab Configuration
- **Runtime**: Python 3.10.x (Colab default)
- **GPU Options**: 
  - T4 (16GB VRAM) - Free tier
  - A100 (40GB VRAM) - Colab Pro
- **Session Limits**: ~12 hours continuous runtime

### Colab Workflow
1. **Upload Code**: Copy modular Python files from local
2. **Upload Dataset**: IDD Lite dataset (~2GB) via drag-drop  
3. **Install Dependencies**: `!pip install -r requirements.txt`
4. **Execute Training**: Full GPU-accelerated training
5. **Download Results**: Models, metrics, visualizations

## Data Management

### Local Dataset Location
```
/Users/harsh/Desktop/RF-Task/data/idd20k_lite/
├── gtFine/
│   ├── train/ (309 dirs with label images)
│   └── val/ (61 dirs with label images)  
└── leftImg8bit/
    ├── train/ (309 dirs with input images)
    └── val/ (61 dirs with input images)
```

### Colab Dataset Upload
- **Method**: Direct upload to Colab `/content/` directory
- **Size**: ~2GB total for IDD Lite dataset
- **Transfer Time**: ~5-10 minutes upload

## Development Workflow

### Phase 1: Local Development
1. **Environment Setup**: Virtual environment + dependencies
2. **Code Development**: Models, data loaders, training scripts in Cursor
3. **Unit Testing**: Test with small data samples locally
4. **Pipeline Validation**: Ensure end-to-end functionality

### Phase 2: Cloud Training  
1. **Colab Setup**: Upload code + dataset
2. **Dependency Installation**: Replicate local environment
3. **Training Execution**: Full dataset training on GPU
4. **Results Collection**: Download trained models + metrics

### Phase 3: Local Analysis
1. **Results Integration**: Bring trained models back to local
2. **Evaluation**: Analysis and visualization locally
3. **Documentation**: Complete project documentation

## Estimated Timeline
- **Local Development**: 1-2 hours
- **Colab Training**: 1-2 hours  
- **Analysis & Documentation**: 30-60 minutes
- **Total Project Time**: 3-5 hours

## Cost Analysis
- **Local Development**: $0 (using existing hardware)
- **Cloud Training**: $0 (Colab free tier sufficient)
- **Total Project Cost**: $0

**Note**: This hybrid approach maximizes development efficiency while minimizing costs and complexity. 