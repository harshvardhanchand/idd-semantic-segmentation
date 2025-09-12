# ⚠️ ARCHIVED: RunPod Configuration 

**Note**: This setup was explored but **not used**. We switched to a **hybrid Cursor (local) + Google Colab (cloud)** approach for better simplicity and cost-effectiveness. See `ENVIRONMENT_SETUP.md` for the actual implementation.

---

# RunPod Configuration for IDD Semantic Segmentation

## Instance Details
- **GPU**: RTX 4000 Ada (20GB VRAM, Ada Lovelace architecture)
- **Cost**: $0.26/hour (~$0.78 total project cost)
- **Container**: PyTorch 2.8.0 + CUDA 12.1 + Ubuntu 22.04

## Storage & Networking
- **Persistent Volume**: 50GB mounted at `/workspace`
- **Container Disk**: 30GB (temporary)
- **Exposed Ports**: 
  - SSH (22): Cursor remote development
  - HTTP (8888): Jupyter notebooks  
  - HTTP (6006): TensorBoard monitoring

## Why This Setup?
- **RTX 4000 Ada**: Perfect balance of performance, VRAM (20GB), and cost for semantic segmentation
- **50GB Volume**: Sufficient for IDD dataset, models, and results with room for experiments
- **PyTorch 2.8.0**: Latest stable version with optimal CUDA 12.1 support
- **Multi-port access**: Seamless development workflow (coding in Cursor + monitoring in browser)

## Development Workflow
1. **Code**: Edit directly via SSH remote development
2. **Train**: Run training scripts on RTX 4000 Ada
3. **Monitor**: View metrics via TensorBoard at `pod-ip:6006`
4. **Experiment**: Use Jupyter notebooks at `pod-ip:8888`

**Estimated Runtime**: 2-3 hours for complete IDD semantic segmentation pipeline. 