"""Model adapters: one uniform interface for every kind of trained classifier.

    adapter.run(rgb_uint8)  ->  {'logits', 'gradcam', 'gradcam++', ['embedding'], ['lesion_gate']}

V21Adapter      models from the v21 notebook (<trunk>_Trunk -> <head>_Head / BaselineHead):
                Grad-CAM on the native 14x14 lesion map, embedding for the feature-OOD gate,
                the hybrid's lesion-attention map (exactly as before).
KerasAdapter    ANY other Keras / tf.keras classifier (.keras or .h5): input size read from the
                model, softmax or logit outputs, dict / list outputs, Grad-CAM on the last
                convolutional (4-D) feature map, or on the ViT token grid, or a gradient
                saliency map when the model has neither.
SavedModelAdapter  TensorFlow SavedModel folders (prediction + input-gradient saliency).
"""
import math

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], np.float32)
_PREFERRED_KEYS = ('pred', 'predictions', 'prediction', 'probs', 'probabilities', 'output', 'outputs',
                   'logits', 'classifier', 'dense', 'output_0')
_RESCALE_LAYERS = ('Rescaling', 'Normalization')


def preprocess(rgb, mode):
    """rgb uint8 (H, W, 3) -> float32 network input for the given training preprocessing."""
    x = rgb.astype(np.float32)
    if mode == '0-255':
        return x
    if mode == '-1-1':
        return x / 127.5 - 1.0
    if mode == 'caffe':                       # keras.applications VGG16/ResNet50 'caffe' preprocessing
        return x[..., ::-1] - CAFFE_MEAN_BGR
    if mode == 'torch':
        return (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x / 255.0                          # '0-1' (v21 notebook, ImageDataGenerator(rescale=1/255))


def cams_from(A, g):
    """Grad-CAM and Grad-CAM++ from activations A (h, w, c) and gradients g (h, w, c)."""
    A, g = np.asarray(A, np.float64), np.asarray(g, np.float64)
    gradcam = np.maximum((A * g.mean(axis=(0, 1))).sum(-1), 0)
    g2, g3 = g ** 2, g ** 3
    denom = 2 * g2 + A.sum(axis=(0, 1), keepdims=True) * g3
    alpha = np.where(np.abs(denom) > 1e-12, g2 / (denom + 1e-12), 0.0)
    gradcampp = np.maximum((A * (alpha * np.maximum(g, 0)).sum(axis=(0, 1))).sum(-1), 0)
    return gradcam, gradcampp


def _shape(t):
    try:
        return tuple(t.shape)
    except Exception:  # noqa: BLE001
        return None


def _token_grid(n):
    """ViT tokens -> (h, w, drop_first): 196 -> (14, 14, 0); 197 (with CLS) -> (14, 14, 1)."""
    for drop in (0, 1):
        s = int(round(math.sqrt(n - drop)))
        if s >= 2 and s * s == n - drop:
            return s, s, drop
    return None


class BaseAdapter:
    arch = 'keras'
    cam_method = 'Grad-CAM'
    cam_layer = None
    cam_hw = None

    def __init__(self, name, model, preprocess_mode='0-1'):
        self.name, self.model = name, model
        self.preprocess_mode = preprocess_mode
        self.input_hw = (224, 224)
        self.channels = 3
        self.num_classes = None
        self.prob_output = False
        self._step = None

    # resize the 224x224 background-removed crop to this model's own input size
    def prepare(self, rgb):
        h, w = self.input_hw
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if h < rgb.shape[0] else cv2.INTER_LINEAR)
        x = preprocess(rgb, self.preprocess_mode)
        if self.channels == 1:
            x = x.mean(-1, keepdims=True)
        return x[None]

    def describe(self):
        h, w = self.input_hw
        bits = [f'input {h}×{w}×{self.channels}', f'{self.num_classes} classes' if self.num_classes else '',
                f'{self.cam_method} on {self.cam_layer}' + (f' ({self.cam_hw[0]}×{self.cam_hw[1]})' if self.cam_hw else '')
                if self.cam_layer else self.cam_method]
        return ' · '.join(b for b in bits if b)

    def info(self):
        return {'arch': self.arch, 'input': list(self.input_hw) + [self.channels], 'num_classes': self.num_classes,
                'cam_method': self.cam_method, 'cam_layer': self.cam_layer,
                'cam_hw': list(self.cam_hw) if self.cam_hw else None,
                'preprocess': self.preprocess_mode, 'outputs': 'probabilities' if self.prob_output else 'logits'}

    @staticmethod
    def _to_logits(y, prob):
        import tensorflow as tf
        y = tf.cast(y, tf.float32)
        if prob:
            if y.shape[-1] == 1:                                   # sigmoid binary output
                p = tf.clip_by_value(y, 1e-7, 1 - 1e-7)
                return tf.concat([tf.zeros_like(p), tf.math.log(p / (1 - p))], -1)
            return tf.math.log(tf.clip_by_value(y, 1e-9, 1.0))
        if y.shape[-1] == 1:
            return tf.concat([tf.zeros_like(y), y], -1)
        return y


class V21Adapter(BaseAdapter):
    """v21 notebook model: Model(img -> head(trunk(img))['pred'])."""
    arch = 'v21'

    def __init__(self, name, model, trunk, head):
        super().__init__(name, model, '0-1')
        self.trunk, self.head = trunk, head
        shp = model.input_shape
        self.input_hw = (int(shp[1] or 224), int(shp[2] or 224))
        self.cam_layer = trunk.name
        self.cam_method = 'Grad-CAM'

    @staticmethod
    def find_parts(model):
        trunk = next((l for l in model.layers if l.name.endswith('_Trunk')), None)
        head = next((l for l in model.layers if l.name.endswith('_Head') or l.name == 'BaselineHead'), None)
        return (trunk, head) if trunk is not None and head is not None else None

    def _build(self):
        import tensorflow as tf
        trunk, head = self.trunk, self.head

        @tf.function(reduce_retracing=True)
        def step(x):
            with tf.GradientTape() as tape:
                lm = trunk(x, training=False)
                tape.watch(lm)
                out = head(lm, training=False)
                logits = tf.cast(out['logits'], tf.float32)
                score = tf.reduce_max(logits[0])
            res = {'logits': logits, 'A': tf.cast(lm, tf.float32), 'grad': tf.cast(tape.gradient(score, lm), tf.float32)}
            if 'embedding' in out:
                res['embedding'] = tf.cast(out['embedding'], tf.float32)
            if 'aux_pseudo_lesion_head' in out:
                res['lesion_gate'] = tf.cast(out['aux_pseudo_lesion_head'], tf.float32)
            return res
        return step

    def run(self, rgb):
        import tensorflow as tf
        if self._step is None:
            self._step = self._build()
        o = self._step(tf.convert_to_tensor(self.prepare(rgb)))
        A, g = o['A'][0].numpy(), o['grad'][0].numpy()
        self.cam_hw = A.shape[:2]
        gc, gpp = cams_from(A, g)
        res = {'logits': o['logits'][0].numpy().astype(np.float64), 'gradcam': gc, 'gradcam++': gpp}
        if self.num_classes is None:
            self.num_classes = int(res['logits'].shape[-1])
        if 'embedding' in o:
            res['embedding'] = o['embedding'][0].numpy().astype(np.float64)
        if 'lesion_gate' in o:
            res['lesion_gate'] = o['lesion_gate'][0, ..., 0].numpy()
        return res


class KerasAdapter(BaseAdapter):
    """Any Keras classifier."""

    def __init__(self, name, model, preprocess_mode='auto'):
        super().__init__(name, model, preprocess_mode)
        import keras
        self.keras = keras
        model = self.src = self._graph_source(model)
        inp = model.inputs if getattr(model, 'inputs', None) else None
        if inp is not None and len(inp) != 1:
            raise ValueError(f'the model has {len(inp)} inputs; only single-image models are supported')
        shp = _shape(inp[0]) if inp else tuple(model.input_shape)
        if shp is None or len(shp) != 4:
            raise ValueError(f'expected an image input (batch, H, W, C), got {shp}')
        self.channels_first = shp[1] in (1, 3) and shp[3] not in (1, 3)
        if self.channels_first:
            shp = (shp[0], shp[2], shp[3], shp[1])
        self.input_hw = (int(shp[1] or 224), int(shp[2] or 224))
        self.channels = int(shp[3] or 3)
        if self.preprocess_mode in (None, '', 'auto'):
            self.preprocess_mode = self._guess_preprocess()
        self._out_tensor, self._out_key = self._pick_output()
        self.num_classes = int(_shape(self._out_tensor)[-1])
        self._unsoftmax()
        self.graph, self._cam_tensor = None, None
        for t, layer_name, method, tokens in self._cam_candidates():
            try:                                   # a nested sub-model's tensor may not be on the outer graph
                self.graph = keras.Model(model.inputs, [self._out_tensor, t])
                self._cam_tensor, self.cam_layer, self.cam_method, self._tokens = t, layer_name, method, tokens
                break
            except Exception:  # noqa: BLE001
                continue
        if self.graph is None:
            self.cam_layer, self.cam_method, self._tokens = None, 'gradient saliency', None
            self.graph = keras.Model(model.inputs, [self._out_tensor])

    # ---------------------------------------------------------------- inspection
    def _graph_source(self, model):
        """A functional graph to read layer tensors from (Sequential models are wrapped)."""
        if model.__class__.__name__ != 'Sequential':
            return model
        f = getattr(model, '_functional', None)
        if f is not None:
            return f
        i = self.keras.Input(tuple(model.input_shape[1:]))
        x = i
        for l in model.layers:
            x = l(x)
        return self.keras.Model(i, x)

    @staticmethod
    def _layer_tensors(l):
        out, seen = [], set()
        for n in list(getattr(l, '_inbound_nodes', []) or [])[::-1]:
            for t in (getattr(n, 'output_tensors', None) or []):
                if id(t) not in seen:
                    seen.add(id(t)); out.append(t)
        if not out:
            try:
                t = l.output
                if not isinstance(t, (list, tuple, dict)):
                    out.append(t)
            except Exception:  # noqa: BLE001
                pass
        return out

    def _all_layers(self, model=None, depth=0):
        model = model or self.src
        for l in model.layers:
            yield l
            if depth < 2 and hasattr(l, 'layers') and l.layers:
                yield from self._all_layers(l, depth + 1)

    def _guess_preprocess(self):
        """A Rescaling / Normalization layer near the input (or a keras.applications EfficientNet /
        ConvNeXt, which scale internally) means the model expects 0-255 pixels; otherwise 0-1."""
        for i, l in enumerate(self._all_layers()):
            if i > 12:
                break
            cls = l.__class__.__name__
            if cls in _RESCALE_LAYERS or any(k in l.name.lower() for k in ('rescaling', 'normalization')) \
                    and 'batch' not in l.name.lower():
                return '0-255'
            if any(k in l.name.lower() for k in ('efficientnet', 'convnext')):
                return '0-255'
        return '0-1'

    def _pick_output(self):
        out = self.src.output
        if isinstance(out, dict):
            for k in _PREFERRED_KEYS:
                if k in out and len(_shape(out[k]) or ()) == 2:
                    return out[k], k
            for k, v in out.items():
                if len(_shape(v) or ()) == 2:
                    return v, k
            raise ValueError('none of the model outputs is a (batch, classes) tensor')
        if isinstance(out, (list, tuple)):
            for i, v in enumerate(out):
                if len(_shape(v) or ()) == 2:
                    return v, i
            raise ValueError('none of the model outputs is a (batch, classes) tensor')
        if len(_shape(out) or ()) != 2:
            raise ValueError(f'the model output has shape {_shape(out)}; expected (batch, classes)')
        return out, None

    def _unsoftmax(self):
        """Use the pre-softmax scores when possible (better Grad-CAM, proper temperature scaling)."""
        keras = self.keras
        out = self._out_tensor
        hist = getattr(out, '_keras_history', None)
        layer = hist[0] if hist else None
        if layer is None:
            return
        act = getattr(layer, 'activation', None)
        act_name = getattr(act, '__name__', str(act)).lower() if act is not None else ''
        cls = layer.__class__.__name__
        if cls == 'Dense' and act_name in ('softmax', 'sigmoid'):
            self.prob_output = act_name == 'sigmoid' and self.num_classes == 1
            if not self.prob_output:
                layer.activation = keras.activations.linear       # our private float32 copy of the model
            return
        if cls in ('Softmax',) or (cls == 'Activation' and act_name == 'softmax'):
            try:
                self._out_tensor = layer.input
                return
            except Exception:  # noqa: BLE001
                pass
        if act_name in ('softmax', 'sigmoid') or cls in ('Softmax',):
            self.prob_output = True

    def _cam_candidates(self):
        """Feature maps for Grad-CAM, best first: last spatial map (batch, h, w, c), then a ViT token
        sequence that forms a square grid. Nothing suitable -> gradient saliency."""
        maps, toks = [], []
        for l in reversed(list(self.src.layers)):
            if l.__class__.__name__ == 'InputLayer':
                continue
            for t in self._layer_tensors(l):
                s = _shape(t)
                if not s:
                    continue
                if len(s) == 4 and not self.channels_first and (s[1] or 0) > 1 and (s[2] or 0) > 1:
                    maps.append((t, l.name, 'Grad-CAM', None))
                elif len(s) == 3 and s[1] and _token_grid(s[1]):
                    toks.append((t, l.name, 'Grad-CAM (ViT tokens)', _token_grid(s[1])))
        return maps[:6] + toks[:3]

    # ---------------------------------------------------------------- running
    def _build(self):
        import tensorflow as tf
        graph, prob, has_cam = self.graph, self.prob_output, self._cam_tensor is not None
        to_logits = self._to_logits

        @tf.function(reduce_retracing=True)
        def step(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                outs = graph(x, training=False)
                outs = outs if isinstance(outs, (list, tuple)) else [outs]
                logits = to_logits(outs[0], prob)
                score = tf.reduce_max(logits[0])
            if has_cam:
                A = tf.cast(outs[1], tf.float32)
                g = tape.gradient(score, outs[1])
                return {'logits': logits, 'A': A, 'grad': tf.cast(g, tf.float32)}
            return {'logits': logits, 'A': tf.cast(x, tf.float32), 'grad': tf.cast(tape.gradient(score, x), tf.float32)}
        return step

    def run(self, rgb):
        import tensorflow as tf
        if self._step is None:
            self._step = self._build()
        x = self.prepare(rgb)
        if self.channels_first:
            x = np.transpose(x, (0, 3, 1, 2))
        o = self._step(tf.convert_to_tensor(x))
        logits = o['logits'][0].numpy().astype(np.float64)
        A, g = o['A'][0].numpy(), o['grad'][0].numpy()
        if self._cam_tensor is None:                        # saliency: |gradient x input|
            if self.channels_first:
                A, g = np.transpose(A, (1, 2, 0)), np.transpose(g, (1, 2, 0))
            sal = np.abs(A * g).sum(-1)
            sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), max(1.0, sal.shape[0] / 56))
            self.cam_hw = sal.shape[:2]
            return {'logits': logits, 'gradcam': sal, 'gradcam++': sal}
        if self._tokens:
            h, w, drop = self._tokens
            A, g = A[drop:].reshape(h, w, -1), g[drop:].reshape(h, w, -1)
        self.cam_hw = A.shape[:2]
        gc, gpp = cams_from(A, g)
        return {'logits': logits, 'gradcam': gc, 'gradcam++': gpp}


class SavedModelAdapter(BaseAdapter):
    """TensorFlow SavedModel folder (Keras 3 cannot load these as Keras models)."""
    arch = 'savedmodel'
    cam_method = 'gradient saliency'

    def __init__(self, name, path, preprocess_mode='0-1'):
        import tensorflow as tf
        sm = tf.saved_model.load(str(path))
        fn = sm.signatures.get('serving_default') if hasattr(sm, 'signatures') else None
        if fn is None:
            raise ValueError('the SavedModel has no serving_default signature')
        super().__init__(name, sm, preprocess_mode if preprocess_mode not in (None, '', 'auto') else '0-1')
        self.fn = fn
        spec = list(fn.structured_input_signature[1].values())
        if len(spec) != 1 or len(spec[0].shape) != 4:
            raise ValueError('only single-image SavedModels are supported')
        self.in_key = list(fn.structured_input_signature[1].keys())[0]
        s = spec[0].shape
        self.input_hw = (int(s[1] or 224), int(s[2] or 224))
        self.channels = int(s[3] or 3)
        self.in_dtype = spec[0].dtype
        outs = {k: v for k, v in fn.structured_outputs.items() if len(v.shape) == 2}
        if not outs:
            raise ValueError('no (batch, classes) output in the SavedModel')
        self.out_key = next((k for k in _PREFERRED_KEYS if k in outs), list(outs)[0])
        self.num_classes = int(outs[self.out_key].shape[-1])
        self.prob_output = None                   # decided from the first output

    def run(self, rgb):
        import tensorflow as tf
        x = tf.convert_to_tensor(self.prepare(rgb))
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.cast(self.fn(**{self.in_key: tf.cast(x, self.in_dtype)})[self.out_key], tf.float32)
            if self.prob_output is None:
                v = y.numpy()[0]
                self.prob_output = bool(v.min() >= 0 and v.max() <= 1 and (abs(v.sum() - 1) < 1e-3 or v.size == 1))
            logits = self._to_logits(y, self.prob_output)
            score = tf.reduce_max(logits[0])
        try:
            g = tape.gradient(score, x)
            sal = np.abs((x * g)[0].numpy()).sum(-1)
            sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), max(1.0, sal.shape[0] / 56))
        except Exception:  # noqa: BLE001  (some exported graphs are not differentiable)
            sal = np.zeros(self.input_hw, np.float32)
        self.cam_hw = sal.shape[:2]
        return {'logits': logits[0].numpy().astype(np.float64), 'gradcam': sal, 'gradcam++': sal}
