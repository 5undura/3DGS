"""Difix3D gsplat 트레이너를 우리 촬영 데이터(복도/방)에 맞게 수정.
1) Difix 모델은 처음 보정할 때만 불러옴 (기본 학습 중 GPU 메모리 절약)
2) 보정할 '새 각도' = 학습 카메라를 좌우로 돌려 벽을 보게 한 각도.
   라운드마다 NC_STEP_DEG 씩 더 돌리고, 최대 NC_MAX_DEG 까지.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[NoClip]" in src:
    print("이미 수정됨"); sys.exit(0)

old_init = '''        # Diffusion fixer
        self.difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        self.difix.set_progress_bar_config(disable=True)
        self.difix.to("cuda")
'''
new_init = '''        # Diffusion fixer (loaded lazily on first fix)
        self.difix = None
'''
old_fix = '''        if len(self.cfg.fix_steps) == 1:
            novel_poses = self.parser.camtoworlds[self.valset.indices]
        else:
            novel_poses = self.interpolator.shift_poses(self.current_novel_poses, self.parser.camtoworlds[self.valset.indices], distance=0.5)
'''
new_fix = '''        if self.difix is None:
            self.difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
            self.difix.set_progress_bar_config(disable=True)
            self.difix.to("cuda")
        # [NoClip] look sideways toward the walls, a little more every round
        k = len(self.novelloaders) + 1
        max_deg = float(os.environ.get("NC_MAX_DEG", 60))
        step_deg = float(os.environ.get("NC_STEP_DEG", 15))
        stride = int(os.environ.get("NC_STRIDE", 8))
        ang = np.deg2rad(min(max_deg, k * step_deg))
        base = self.parser.camtoworlds[np.array(self.trainset.indices)[::stride]]
        rotated = []
        for sgn in (1.0, -1.0):
            c, s = np.cos(sgn * ang), np.sin(sgn * ang)
            Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            p = base.copy()
            p[:, :3, :3] = base[:, :3, :3] @ Ry  # yaw around the camera's own up/down axis
            rotated.append(p)
        novel_poses = np.concatenate(rotated, 0)
        print(f"[NoClip] fix round {k}: {len(novel_poses)} views, yaw +-{np.rad2deg(ang):.0f} deg")
'''
old_alpha = '''        parser.alpha_mask_paths = [f"{self.render_dir}/novel/{step}/Alpha/{i:04d}.png" for i in range(len(novel_poses))]
'''
new_alpha = '''        parser.alpha_mask_paths = [f"{self.render_dir}/novel/{step}/Alpha/{i:04d}.png" for i in range(len(novel_poses))]
        if os.environ.get("NC_NO_ALPHA") == "1":  # [NoClip] also learn where the render was empty
            parser.alpha_mask_paths = None
'''
for o, n in [(old_init, new_init), (old_fix, new_fix), (old_alpha, new_alpha)]:
    assert src.count(o) == 1, "패치 위치를 찾지 못함: " + o[:60]
    src = src.replace(o, n)
open(path, "w", encoding="utf-8").write(src)
print("수정 완료:", path)
