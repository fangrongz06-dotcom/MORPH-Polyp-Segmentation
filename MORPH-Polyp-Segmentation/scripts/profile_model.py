"""FP32 0-m profiling: MAC=1 FLOP, 20 warmups, 100 timed forwards."""
import argparse
import json
import sys
import time
import types
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import MORPH
from utils.common import load_config, resolve_config, select_device


class Logits(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
    def forward(self,image):
        return self.model(image)["logits"]


def count_complexity(model,image):
    from fvcore.nn import FlopCountAnalysis
    wavelet_macs = 0
    original = None
    if model.whfc is not None:
        whfc = model.whfc
        if whfc.wavelet!="haar":
            raise ValueError("The analytic fixed-DWT profiling handler currently supports Haar only")
        basis = whfc(image)[1]["basis"]
        original = whfc.forward
        # symmetric DWT uses NumPy shape arithmetic which cannot be JIT-traced.
        # Trace its learned projections with precomputed coefficients, then add
        # separable two-tap Haar filtering MACs. Restore real DWT before timing.
        def cached_forward(self,value,size=(64,64)):
            return tuple(p(basis) for p in self.projections), {"basis":basis,"details":[]}
        whfc.forward = types.MethodType(cached_forward,whfc)
        height,width = image.shape[-2:]
        for _ in range(whfc.levels):
            if height%2 or width%2: raise ValueError("Haar MAC handler requires even dimensions at each scale")
            wavelet_macs += int(image.shape[0])*4*3*height*width
            height,width = height//2,width//2
    try:
        analysis = FlopCountAnalysis(Logits(model),(image,))
        count = analysis.total()
        unsupported = dict(analysis.unsupported_ops())
    finally:
        if original is not None: model.whfc.forward=original
    return count+wavelet_macs,unsupported,wavelet_macs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"configs/morph_0m.yaml"))
    parser.add_argument("--sam_checkpoint",required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device",choices=["cpu","cuda","auto"],default="cuda")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--count_only",action="store_true",help="Count complexity without latency or GPU memory measurement")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output exists")
    config = resolve_config(load_config(args.config), ROOT, sam_checkpoint=args.sam_checkpoint)
    config["model"]["activation_checkpointing"] = False
    device = select_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MORPH(config).float().to(device).eval()
    if args.checkpoint:
        payload = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        model.load_state_dict(payload["model"],strict=True)
    forward = Logits(model)
    image = torch.zeros(1,3,1024,1024,device=device)
    with torch.no_grad():
        count,unsupported,wavelet_macs = count_complexity(model,image)
        for _ in range(0 if args.count_only else 20):
            forward(image)
        if device.type=="cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(0 if args.count_only else 100):
            if device.type=="cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            output = forward(image)
            if device.type=="cuda": torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-start)*1000)
            del output
    latency = sum(elapsed)/len(elapsed) if elapsed else None
    report = {"pytorch":str(torch.__version__),"cuda":torch.version.cuda,
              "device":torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
              "precision":"FP32","protocol":"0-m","input_shape":[1,3,1024,1024],
              "warmup_iterations":0 if args.count_only else 20,"timed_iterations":len(elapsed),"MAC_to_FLOP":1,
              "total_parameters":sum(p.numel() for p in model.parameters()),
              "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
              "profiled_flops":count,"unsupported_operations":unsupported,
              "fixed_Haar_filter_MACs":wavelet_macs,"flops_backend":"fvcore plus analytic separable two-tap Haar filtering",
              "flop_count_complete":not unsupported,"latency_ms":latency,"throughput_fps":1000/latency if latency else None,
              "peak_memory_bytes_allocated":torch.cuda.max_memory_allocated(device) if device.type=="cuda" and not args.count_only else None}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
