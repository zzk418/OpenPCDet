#!/usr/bin/env python3
"""QAT-KD 训练: 全精度教师蒸馏 伪量化学生, 恢复 RKNN int8 conf 精度。

背景: rknn 2.3.2 PTQ int8 conf 偏 0.82~0.84 (fp32 0.92~0.96), 是 w8a8 网络级累积误差。
本脚本用 "蒸馏 + 量化感知训练" 组合恢复:
  - 学生 = best.pt 架构, fuse() 折叠 BN (与部署一致), 每个 nn.Conv2d -> FakeQuantConv2d
  - 伪量化方案与 rknn 转换设置对齐:
      * 权重 per-channel int8 对称 [-127,127] (对应转换 --method channel)
      * 激活 per-tensor int8 对称, 静态 scale = 训练集标定 max (对应 rknn 静态标定)
  - 教师 = best.pt 冻结 fp32, 输出解码 y [1,11,8400] 作为软目标
  - loss = 任务 loss + λ·KD(s_y, t_y)

训练完 unwrap -> 干净权重 -> ultralytics 导出 ONNX -> 现有 onnx_scale_conf + export_rknn
(--method channel) 管线转换验证。

环境: conda activate pc (GPU); 用法:
    cd /code/OpenPCDet && python tools/qat_kd_train.py [--epochs 12] [--batch 8]
"""
import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

IMGSZ = 640
BASE = 'output/shelf_pose_train/shelf_v6_s_night_c2/weights/best.pt'
DATA_YAML = 'datasets/shelf_pose_reviewed/dataset.yaml'
DATA_PATH = 'datasets/shelf_pose_reviewed'


# ---------- 伪量化 (STE) ----------
class FakeQuantConv2d(nn.Module):
    """模拟 int8 部署的 conv, 与 rknn 转换对齐:
      - 权重 per-channel 对称 int8 (scale 由当前权重动态求, 与 rknn 按最终权重定 scale 一致)
      - 激活 per-tensor 对称 int8, scale = calibrate() 标定的静态值 (与 rknn 静态标定一致)
    """

    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.register_buffer('act_scale', torch.tensor(1.0))
        self._cal_max = None  # 标定时累积的激活最大值

    @staticmethod
    def _ste_round(x):
        """STE 取整: 前向 = round(x), 反向 = 直通梯度 (torch 2.0.1 的 round backward 是 0, 会杀梯度)."""
        return (x.round() - x).detach() + x

    def _act_quant(self, x):
        s = self.act_scale
        return self._ste_round(x / s).clamp(-127, 127) * s

    def _weight_quant(self, w):
        wmax = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        wscale = wmax / 127.0
        return self._ste_round(w / wscale).clamp(-127, 127) * wscale

    def forward(self, x):
        w = self.conv.weight
        if self.training and getattr(self, 'calibrating', False):
            # 标定模式: 只记录激活 max, 不量化
            m = x.detach().abs().amax()
            self._cal_max = m if self._cal_max is None else torch.maximum(self._cal_max, m)
            return F.conv2d(x, w, self.conv.bias, self.conv.stride,
                            self.conv.padding, self.conv.dilation, self.conv.groups)
        return F.conv2d(self._act_quant(x), self._weight_quant(w), self.conv.bias,
                        self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)


def wrap_fake_quant(model):
    n = 0
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(model, name, FakeQuantConv2d(child))
            n += 1
        else:
            n += wrap_fake_quant(child)
    return n


def unwrap_fake_quant(model):
    n = 0
    for name, child in model.named_children():
        if isinstance(child, FakeQuantConv2d):
            setattr(model, name, child.conv)
            n += 1
        else:
            n += unwrap_fake_quant(child)
    return n


def calibrate_act_scales(model, device, n_calib=100):
    """用训练集标定每个 conv 的静态激活 scale。

    与 rknn 转换完全一致: 前 n_calib 张 train 图, manual letterbox 640+fill114
    (不用 ultralytics loader 的预处理, 两者 letterbox 有差异会导致 scale 偏乐观)。
    """
    import cv2
    import numpy as np
    model.train()
    for m in model.modules():
        if isinstance(m, FakeQuantConv2d):
            m.calibrating = True
            m._cal_max = None
    imgs = sorted(glob.glob(os.path.join(DATA_PATH, 'images/train', '*.jpg')))
    imgs = imgs[:n_calib]
    for p in imgs:
        im = cv2.imread(p)
        r = min(IMGSZ / im.shape[0], IMGSZ / im.shape[1])
        nu = (int(round(im.shape[1] * r)), int(round(im.shape[0] * r)))
        im2 = cv2.resize(im, nu)
        top, left = (IMGSZ - nu[1]) // 2, (IMGSZ - nu[0]) // 2
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        canvas[top:top + nu[1], left:left + nu[0]] = im2
        x = torch.from_numpy(canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device)
        with torch.no_grad():
            model(x)  # 训练前向, 只记录激活 max, 不量化
    for m in model.modules():
        if isinstance(m, FakeQuantConv2d):
            m.calibrating = False
            if m._cal_max is not None:
                m.act_scale.fill_((m._cal_max / 127.0).item())
            # None = 该 conv 未在可执行路径上 (如 end2end 分支), 保持默认 scale, 不影响
    print(f'激活标定完成: {len(imgs)} 张 (manual letterbox, 与 rknn 一致)')


# ---------- 数据 ----------
def build_loader(batch, shuffle=True):
    from ultralytics.data import build_dataloader
    from ultralytics.data.dataset import YOLODataset
    data = {'yaml_file': DATA_YAML, 'path': DATA_PATH, 'train': 'images/train',
            'val': 'images/val', 'kpt_shape': [2, 3], 'flip_idx': [0, 1], 'names': {0: 'shelf'}}
    ds = YOLODataset(os.path.join(DATA_PATH, 'images/train'), imgsz=IMGSZ,
                     batch_size=batch, augment=False, rect=False, data=data, task='pose')
    return build_dataloader(ds, batch=batch, workers=4, shuffle=shuffle)


def to_device(batch, device):
    out = dict(batch)
    out['img'] = batch['img'].to(device, non_blocking=True).float() / 255.0
    for k in ('bboxes', 'cls', 'keypoints', 'batch_idx'):
        out[k] = batch[k].to(device, non_blocking=True)
    return out


def measure_conf(model, device):
    """伪量化学生 (≈模拟 int8) 在 5 张验证图上的 conf, 用于验证模拟忠实度与训练进展。"""
    import cv2
    import numpy as np
    model.eval()
    lines = []
    for p in sorted(glob.glob('output/rk3588_deploy/imgs/*.jpg')):
        im = cv2.imread(p)
        r = min(IMGSZ / im.shape[0], IMGSZ / im.shape[1])
        nu = (int(round(im.shape[1] * r)), int(round(im.shape[0] * r)))
        im2 = cv2.resize(im, nu)
        top, left = (IMGSZ - nu[1]) // 2, (IMGSZ - nu[0]) // 2
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        canvas[top:top + nu[1], left:left + nu[0]] = im2
        x = torch.from_numpy(canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device)
        with torch.no_grad():
            y = model(x)[0]
        conf = y[0, 4]
        c = conf[conf >= 0.25]
        top_c = float(c.max()) if c.numel() else 0.0
        lines.append(f'{os.path.basename(p).replace(".jpg", "")}={top_c:.3f}')
    return ' | '.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--kd-lambda', type=float, default=0.5)
    ap.add_argument('--kd-conf-lambda', type=float, default=1.0, help='conf 通道专属 KD 权重')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='output/shelf_pose_train/shelf_v6_s_night_c2/weights/qat_best.pt')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')

    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics import YOLO

    hyp = dict(box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64,
               angle=1.0, dlog=1.0, dgrad=0.5, dlam=1.0)

    # 教师: 冻结 fp32 (fuse 与部署一致)
    teacher_yolo = YOLO(BASE)
    teacher = teacher_yolo.model.to(device)
    teacher.args = IterableSimpleNamespace(**hyp)
    teacher = teacher.fuse().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 学生: 同架构, fuse, 伪量化
    student_yolo = YOLO(BASE)
    student = student_yolo.model.to(device)
    student.args = IterableSimpleNamespace(**hyp)
    student = student.fuse().train()
    for p in student.parameters():
        p.requires_grad_(True)  # best.pt 存的是冻结参数, 需解冻
    n_q = wrap_fake_quant(student)
    print(f'学生伪量化 conv 数: {n_q}')

    loader = build_loader(args.batch, shuffle=True)
    # 激活静态标定 (模拟 rknn 标定, manual letterbox)
    calibrate_act_scales(student, device, n_calib=100)

    print('训练前 模拟int8 conf (应与 rknn PTQ 0.82/0.82/0.34 接近, 证明模拟忠实):')
    print(' ', measure_conf(student, device))

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=5e-4)
    steps_per_epoch = len(loader)
    for ep in range(args.epochs):
        student.train()
        tot_loss = tot_kd = tot_kdc = tot_n = 0.0
        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            loss, _ = student(batch)  # 任务 loss (train 前向, 5 维向量)
            loss = loss.sum()

            student.eval()
            s_y = student(batch['img'])[0]
            with torch.no_grad():
                t_y = teacher(batch['img'])[0]
            student.train()

            kd = F.mse_loss(s_y, t_y)                       # 全通道 KD (box 坐标主导)
            kd_conf = F.mse_loss(s_y[:, 4:5], t_y[:, 4:5])  # 仅 conf 通道: 直接给教师 conf 信号
            scale = (loss.detach() / (kd.detach() + 1e-5)).clamp(0.01, 100)
            cscale = (loss.detach() / (kd_conf.detach() + 1e-5)).clamp(0.01, 100)
            total = loss + args.kd_lambda * scale * kd + args.kd_conf_lambda * cscale * kd_conf

            opt.zero_grad()
            total.backward()
            opt.step()

            tot_loss += float(loss.detach())
            tot_kd += float(kd.detach())
            tot_kdc += float(kd_conf.detach())
            tot_n += 1
            if i % 5 == 0:
                print(f'  ep{ep} step{i}/{steps_per_epoch} loss={loss.item():.2f} '
                      f'kd={kd.item():.2e} kdc={kd_conf.item():.2e}')
        print(f'== ep{ep} 平均 loss={tot_loss/tot_n:.2f} kd={tot_kd/tot_n:.2e} kd_conf={tot_kdc/tot_n:.2e}')
        print(f'   模拟int8 conf: {measure_conf(student, device)}')

    unwrap_fake_quant(student)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'model': student_yolo.model.state_dict()}, args.out)
    print(f'QAT 训练完成, 干净权重 -> {args.out}')


if __name__ == '__main__':
    main()
