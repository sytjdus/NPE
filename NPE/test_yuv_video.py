import argparse
import json
import os
import struct
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.utils.model_zoo import tqdm
from pytorch_wavelets import DWT, IDWT
from HM_datasets.rawvideo import RawVideoSequence_HM, VideoFormat
from utils import convert_yuv420_ycbcr, convert_ycbcr_yuv420, to_tensors, write_frame, pad, crop


Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]
RAWVIDEO_EXTENSIONS = (".yuv",)  # read raw yuv videos for now


def collect_videos(rootpath: str) -> List[str]:
    video_files = []
    for ext in RAWVIDEO_EXTENSIONS:
        video_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(video_files)

def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    metrics = defaultdict(list)

    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data["results"].items():
            metrics[k].append(v)

    # normalize
    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate a video compression network on a video dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, help="sequences directory")
    parser.add_argument("--output", type=str, required=True, help="output directory")
    parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite previous runs"
    )
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="output json file name",
    )
    parser.add_argument(
        "-c",
        "--test_checkpoint",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    parser.add_argument(
        "--only_yuv", action="store_true", help="generate ONLY .yuv output files"
    )
    parser.add_argument(f'--block', type=int, default=16, help='block size for inference')

    def set_model_args(parser):
        group = parser.add_argument_group('specific parameters')
        # LL fusion stages for Y channel
        group.add_argument('--in_chans', type=int, nargs='+', default=[16], help='input channels')
        group.add_argument('--channels', type=int, nargs='+', default=[64, 64], help='channels')
        group.add_argument('--blocks', type=int, nargs='+', default=[2, 2], help='blocks')
        group.add_argument('--dilations', type=int, nargs='+', default=[2], help='dilation')
        
        return parser


    set_model_args(parser)

    args = parser.parse_args(argv)
    return args

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1

def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape

def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_checkpoint(net: nn.Module, checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint, strict=False)
    net.load_state_dict(checkpoint["state_dict"])
    net.load_state_dict(checkpoint, strict=False)
    

    net.eval()
    return net

def compute_metrics_for_frame(
    org_frame: Frame,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # YCbCr metrics
    org_yuv = to_tensors(org_frame, device=str(device), max_value=max_val)
    org_yuv = tuple(p.unsqueeze(0).unsqueeze(0) for p in org_yuv)  # type: ignore
    rec_yuv = convert_ycbcr_yuv420(rec_frame)
    for i, component in enumerate("yuv"):
        org = (org_yuv[i] * max_val).clamp(0, max_val).round()
        rec = (rec_yuv[i] * max_val).clamp(0, max_val).round()
        out[f"psnr-{component}"] = 20 * np.log10(max_val) - 10 * torch.log10(
            (org - rec).pow(2).mean()
        )
    out["psnr-yuv"] = (4 * out["psnr-y"] + out["psnr-u"] + out["psnr-v"]) / 6

    # RGB metrics
    org_rgb = convert_yuv420_ycbcr(
        org_frame, device, max_val
    )  # ycbcr2rgb(yuv_420_to_444(org_frame, mode="bicubic"))  # type: ignore
    org_rgb = (org_rgb * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_rgb - rec_frame).pow(2).mean()
    psnr_rgb = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)

    ms_ssim_rgb = ms_ssim(org_rgb, rec_frame, data_range=max_val)
    out.update({"ms-ssim-rgb": ms_ssim_rgb, "mse-rgb": mse_rgb, "psnr-rgb": psnr_rgb})

    return out

@torch.no_grad()
def eval_model(
    net: nn.Module, sequence: Path, savepath: Path, E2E: bool,
) -> Dict[str, Any]:
    org_seq = RawVideoSequence_HM.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(net.parameters()).device
    num_frames = len(org_seq)
    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)

    f = savepath.open("wb")

    print(f" encoding {sequence.stem}", file=sys.stderr)

    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            x_cur = convert_yuv420_ycbcr(org_seq[i], device, max_val)
            x_cur, padding = pad(x_cur, p=16)

            dwt = DWT(J=1, wave='haar', mode='periodization').to(device)
            idwt = IDWT(wave='haar', mode='periodization').to(device)
            y, uv = x_cur[:, 0:1, :, :], x_cur[:, 1:, :, :]  
            yl, yh = dwt(y)  
            LH = yh[0][:, :, 0, :, :]
            HL = yh[0][:, :, 1, :, :]
            HH = yh[0][:, :, 2, :, :]
            yh_components = torch.cat([LH, HL, HH], dim=1)  # [B,3,H/2,W/2]

            ll_feat, yh_feat = net(yl, yh_components, uv)
            idwt_out = idwt([ll_feat, [yh_feat]])
            NPE_out = torch.cat([idwt_out, uv], dim=1)
            NPE_out = NPE_out.clamp(0, 1)


            NPE_out = crop(NPE_out, padding)
            metrics = compute_metrics_for_frame(
                org_seq[i],
                NPE_out,
                device,
                max_val,
            )

            NPE_out = convert_ycbcr_yuv420(NPE_out)
            write_frame(f, NPE_out, org_seq.bitdepth)

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)
    f.close()

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results

def run_inference(
    filepaths,
    inputdir: Path,
    net: nn.Module,
    outputdir: Path,
    force: bool = False,
    entropy_estimation: bool = False,
    only_yuv: bool = False,
    E2E: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any,
) -> Dict[str, Any]:
    results_paths = []

    for filepath in filepaths:
        output_subdir = Path(outputdir) / Path(filepath).parent.relative_to(inputdir)
        output_subdir.mkdir(parents=True, exist_ok=True)
        sequence_metrics_path = output_subdir / f"{filepath.stem}-{trained_net}.json"
        results_paths.append(sequence_metrics_path)

        if force:
            sequence_metrics_path.unlink(missing_ok=True)
        if sequence_metrics_path.is_file():
            continue

        with torch.no_grad():
            sequence_yuv = output_subdir / f"{filepath.stem}.yuv"
            metrics = eval_model(
                net, filepath, sequence_yuv, E2E,
            )
        
        output = {
            "source": filepath.stem,
            "name": 'NPE',
            "description": f"Inference ({description})",
            "results": metrics,
        }
        
        if not only_yuv:
            with sequence_metrics_path.open("wb") as f:
                f.write(json.dumps(output, indent=2).encode())
        print(json.dumps(output, indent=2))
        
    results = aggregate_results(results_paths) if not only_yuv else None
    return results

def main(args: Any = None) -> None:
    
    def load_model(pt_path: str, device: torch.device = "cpu") -> nn.Module:
        model = torch.jit.load(pt_path, map_location=device)
        model.eval()
        return model

    args = parse_args(args)

    if not args.test_checkpoint:
        print("Error: missing 'test_checkpoint'.", file=sys.stderr)
        raise SystemExit(1)
    
    filepaths = collect_videos(args.dataset)
    if len(filepaths) == 0:
        print("Error: no video found in directory.", file=sys.stderr)
        raise SystemExit(1)

    # create output directory
    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model= load_model("./NPE_model.pt")



    runs = args.test_checkpoint
    description = "NPE"

    results = defaultdict(list)
    for run in runs:
        
        if not os.path.isfile(run):
            print(f"Error: missing {run} file", file=sys.stderr)
            raise SystemExit(1)
        
        cpt_name = Path(run).name[: -len(".tar.pth")]  # removesuffix() python3.9
        trained_net = f"{cpt_name}-{description}"

        print(f"Loading from {run}")
        model = load_checkpoint(model, run)
        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available():
            model = model.to(args.device)
        args_dict = vars(args)
        metrics = run_inference(
            filepaths,
            args.dataset,
            model,
            outputdir,
            trained_net=trained_net,
            description=description,
            **args_dict,
        )
        results["q"].append(trained_net)
        
        if metrics:
            for k, v in metrics.items():
                results[k].append(v)

    output = {
        "name": f"NPE",
        "description": f"Inference ({description})",
        "results": results,
    }

    if args.output_file == "":
        output_file = f"NPE-{description}"
    else:
        output_file = args.output_file

    if not args.only_yuv:
        with (Path(f"{outputdir}/{output_file}").with_suffix(".json")).open("wb") as f:
            f.write(json.dumps(output, indent=2).encode())

if __name__ == "__main__":
    main(sys.argv[1:])


def load_model(pt_path: str, device: torch.device = "cpu") -> nn.Module:
    model = torch.jit.load(pt_path, map_location=device)
    model.eval()
    return model

def set_model_args(parser):
    group = parser.add_argument_group('specific parameters')
    # LL fusion stages for Y channel
    group.add_argument('--in_chans', type=int, nargs='+', default=[16], help='input channels')
    group.add_argument('--channels', type=int, nargs='+', default=[64, 64], help='channels')
    group.add_argument('--blocks', type=int, nargs='+', default=[2, 2], help='blocks')
    group.add_argument('--dilations', type=int, nargs='+', default=[2], help='dilation')
    
    return parser

