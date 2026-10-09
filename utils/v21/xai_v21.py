# CELL 10.1b - Explainability v21: Grad-CAM / Grad-CAM++ / LayerCAM on the native 14x14
# lesion map, honest Leaf-Focus Score (LFS) + Lesion-Focus Score (LeFS) + pointing game,
# and a disease-region view.
#
# WHY GRAD-CAM LANDED OUTSIDE THE LEAF IN v9/v20 (verified in the code):
#   * target map was 7x7 (stride 32) bicubic-upsampled to 14x14, then
#     percentile-stretched, Gaussian-denoised and cubic-resized: three
#     smoothing steps on a 32-px grid -> round blobs that overflow the leaf
#     edge (v9 Cell 10.2 figure);
#   * part of the decision bypassed that map, so the map never explained the
#     whole prediction;
#   * the only "LFS" term in training was an aux head (not Grad-CAM itself).
# V21: native 14x14 map (Cell 5.1b), all evidence through it, the CAM itself
# regularised during training (Cell 7.1b), and here: max-normalised CAM,
# bilinear resize only, no percentile stretch, no blur.
#
# HONESTY RULES baked into the outputs:
#   * lfs / lefs / pointing are ALWAYS computed on the RAW CAM (the model's
#     genuine attention). The leaf-restricted and lesion-guided panels are
#     DISPLAY aids and are labelled as such.
#   * "lesion" = colour-abnormality proxy (V3 segmentation's lesion mask +
#     HSV/Lab proxy), NOT expert pixel annotation.
import cv2
import numpy as np
import tensorflow as tf


def _find_trunk_head(model):
    trunk = next(l for l in model.layers if l.name.endswith('_Trunk'))
    head = next(l for l in model.layers if l.name.endswith('_Head') or l.name == 'BaselineHead')
    return trunk, head


def compute_cams(model, x01, cls=None):
    """x01: (H,W,3) float in [0,1] (the preprocessed, background-removed
    crop). Returns dict of raw (h,w) maps + class/probabilities."""
    trunk, head = _find_trunk_head(model)
    x = tf.convert_to_tensor(x01[None].astype(np.float32))
    with tf.GradientTape() as t:
        lm = trunk(x, training=False)
        t.watch(lm)
        out = head(lm, training=False)
        logits = tf.cast(out['logits'], tf.float32)
        probs = tf.nn.softmax(logits)[0].numpy()
        c = int(np.argmax(probs)) if cls is None else int(cls)
        score = logits[0, c]
    g = tf.cast(t.gradient(score, lm), tf.float32)[0].numpy()
    A = tf.cast(lm, tf.float32)[0].numpy()
    w = g.mean(axis=(0, 1))
    gradcam = np.maximum((A * w).sum(-1), 0)
    g2, g3 = g ** 2, g ** 3
    denom = 2 * g2 + A.sum(axis=(0, 1), keepdims=True) * g3
    alpha = np.where(np.abs(denom) > 1e-12, g2 / (denom + 1e-12), 0.0)
    w_pp = (alpha * np.maximum(g, 0)).sum(axis=(0, 1))
    gradcampp = np.maximum((A * w_pp).sum(-1), 0)
    layercam = np.maximum((np.maximum(g, 0) * A).sum(-1), 0)
    extra = {}
    if 'aux_pseudo_lesion_head' in out:
        extra['lesion_gate'] = tf.cast(out['aux_pseudo_lesion_head'], tf.float32)[0, ..., 0].numpy()
    return {'gradcam': gradcam, 'gradcam++': gradcampp, 'layercam': layercam,
            'cls': c, 'probs': probs, **extra}


def upsample(cam, hw):
    m = float(cam.max())
    cam = cam / m if m > 0 else cam
    return np.clip(cv2.resize(cam.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR), 0, 1)


def lesion_proxy_mask(rgb_uint8, leaf_mask_u8, seg_lesion_u8=None, thresh=0.35):
    """Binary lesion-proxy (union of V3's lesion mask and an HSV/Lab colour
    abnormality proxy inside the leaf). NOT ground truth."""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_d = np.where(h < 35, 35 - h, np.where(h > 85, h - 85, 0.0))
    abn = 0.45 * np.clip(hue_d / 20.0, 0, 1) * (s > 45) + 0.30 * np.clip((lab[..., 1] - 128) / 25, 0, 1) \
        + 0.25 * np.clip((lab[..., 2] - 128 - 25) / 25, 0, 1)
    abn = np.maximum(abn, np.clip((70 - v) / 40, 0, 1) * 0.8)
    leaf = leaf_mask_u8 > 0
    m = (abn >= thresh) & leaf
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if seg_lesion_u8 is not None:
        m |= (seg_lesion_u8 > 0) & leaf
    return m


def focus_metrics(cam_full, leaf_mask_u8, lesion_mask_bool=None):
    """All on the RAW (max-normalised, bilinear) CAM."""
    leaf = leaf_mask_u8 > 0
    tot = float(cam_full.sum()) + 1e-8
    r = {'lfs': float((cam_full * leaf).sum() / tot)}
    yx = np.unravel_index(np.argmax(cam_full), cam_full.shape)
    r['pointing_leaf'] = bool(leaf[yx])
    top = cam_full >= np.quantile(cam_full, 0.90)
    r['top10_in_leaf'] = float((top & leaf).sum() / max(1, top.sum()))
    if lesion_mask_bool is not None and lesion_mask_bool.sum() > 20:
        les = lesion_mask_bool
        r['lefs'] = float((cam_full * les).sum() / tot)
        r['lesion_area_frac_of_leaf'] = float(les.sum() / max(1, leaf.sum()))
        # >1 means the CAM concentrates on lesions more than a uniform-on-leaf map would
        r['lesion_enrichment'] = float((r['lefs'] / max(r['lfs'], 1e-6)) / max(r['lesion_area_frac_of_leaf'], 1e-6))
        dil = cv2.dilate(les.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        r['pointing_lesion'] = bool(dil[yx])
        r['top10_in_lesion'] = float((top & dil).sum() / max(1, top.sum()))
    else:
        r.update({'lefs': None, 'lesion_area_frac_of_leaf': 0.0, 'lesion_enrichment': None,
                  'pointing_lesion': None, 'top10_in_lesion': None})
    return r


def disease_regions(cam_full, lesion_mask_bool, min_area=25, top_k=5):
    """Lesion components ranked by the CAM mass they receive -> boxes of the
    regions the model actually used as disease evidence."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lesion_mask_bool.astype(np.uint8), 8)
    regs = []
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < min_area:
            continue
        mass = float(cam_full[lab == j].sum())
        x, y, w, h = stats[j, :4]
        regs.append({'box': (int(x), int(y), int(w), int(h)), 'area': int(stats[j, 4]), 'cam_mass': mass})
    tot = sum(r['cam_mass'] for r in regs) + 1e-8
    for r in regs:
        r['cam_share'] = r['cam_mass'] / tot
    return sorted(regs, key=lambda r: -r['cam_mass'])[:top_k]


def overlay(rgb, heat, alpha=0.45):
    col = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return np.clip((1 - alpha) * rgb + alpha * col, 0, 255).astype(np.uint8)


def explain_image(model, seg_result, class_names=None, method='gradcam', healthy_names=('Tomato___healthy',)):
    """seg_result: BackgroundAwarePreprocessorV3.process(...) output.
    Returns maps, metrics, disease regions, and 4 display panels."""
    rgb = cv2.cvtColor(seg_result['background_removed'], cv2.COLOR_BGR2RGB)
    leaf_u8 = seg_result['leaf_mask']
    maps = compute_cams(model, rgb.astype(np.float32) / 255.0)
    cam = upsample(maps[method], rgb.shape[:2])
    les = lesion_proxy_mask(rgb, leaf_u8, seg_result.get('lesion_mask'))
    cname = class_names[maps['cls']] if class_names is not None else str(maps['cls'])
    healthy = cname in healthy_names
    met = focus_metrics(cam, leaf_u8, None if healthy else les)
    leaf_cam = cam * (leaf_u8 > 0)
    guided = leaf_cam * (0.25 + 0.75 * cv2.GaussianBlur(les.astype(np.float32), (0, 0), 2))
    guided = guided / (guided.max() + 1e-8)
    regs = [] if healthy else disease_regions(cam, les)
    boxed = overlay(rgb, guided)
    for r in regs[:3]:
        x, y, w, h = r['box']
        cv2.rectangle(boxed, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 255), 2)
    cnts, _ = cv2.findContours(les.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not healthy:
        cv2.drawContours(boxed, cnts, -1, (255, 0, 60), 1)
    return {'class': cname, 'class_index': maps['cls'], 'probs': maps['probs'], 'cam_raw_full': cam,
            'metrics': met, 'disease_regions': regs, 'lesion_proxy': les,
            'panels': {'input': rgb, 'raw_cam (model attention)': overlay(rgb, cam),
                       'leaf-restricted CAM (display)': overlay(rgb, leaf_cam / (leaf_cam.max() + 1e-8)),
                       'disease regions (CAM x lesion proxy)': boxed}}
