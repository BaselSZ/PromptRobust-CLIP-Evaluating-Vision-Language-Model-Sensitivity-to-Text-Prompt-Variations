import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
PET_ATTRIBUTES = {
    "abyssinian": "a short-haired cat with large ears",
    "bengal": "a spotted short-haired cat",
    "birman": "a fluffy cat with blue eyes",
    "bombay": "a black short-haired cat",
    "british shorthair": "a round-faced short-haired cat",
    "egyptian mau": "a spotted short-haired cat",
    "maine coon": "a large fluffy long-haired cat",
    "persian": "a fluffy long-haired flat-faced cat",
    "ragdoll": "a fluffy cat with blue eyes",
    "russian blue": "a gray short-haired cat",
    "siamese": "a slender cat with dark face and ears",
    "sphynx": "a hairless cat",

    "american bulldog": "a muscular short-haired dog",
    "american pit bull terrier": "a muscular short-haired dog",
    "basset hound": "a short-legged dog with long ears",
    "beagle": "a small hound dog with floppy ears",
    "boxer": "a muscular short-haired dog",
    "chihuahua": "a very small dog with large ears",
    "english cocker spaniel": "a dog with long ears and wavy fur",
    "english setter": "a spotted long-haired dog",
    "german shorthaired": "a spotted short-haired dog",
    "great pyrenees": "a large white fluffy dog",
    "havanese": "a small fluffy long-haired dog",
    "japanese chin": "a small long-haired dog with flat face",
    "keeshond": "a fluffy gray dog",
    "leonberger": "a large long-haired brown dog",
    "miniature pinscher": "a small short-haired dog",
    "newfoundland": "a large fluffy dog",
    "pomeranian": "a small fluffy dog",
    "pug": "a small wrinkled flat-faced dog",
    "saint bernard": "a large brown and white dog",
    "samoyed": "a white fluffy dog",
    "scottish terrier": "a small black wiry-haired dog",
    "shiba inu": "a small fox-like dog",
    "staffordshire bull terrier": "a muscular short-haired dog",
    "wheaten terrier": "a soft-coated light-colored dog",
    "yorkshire terrier": "a small long-haired dog",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        )

        # Two different textual initializations
        ctx_init_1 = "a photo of a"
        ctx_init_2 = "a blurry photo of a"

        prompt_1 = clip.tokenize(ctx_init_1)
        prompt_2 = clip.tokenize(ctx_init_2)

        with torch.no_grad():
            embedding_1 = clip_model.token_embedding(prompt_1).type(dtype)
            embedding_2 = clip_model.token_embedding(prompt_2).type(dtype)

        ctx_vectors_1 = embedding_1[0, 1:1 + n_ctx, :]
        ctx_vectors_2 = embedding_2[0, 1:1 + n_ctx, :]

        print(f'Initial context 1: "{ctx_init_1}"')
        print(f'Initial context 2: "{ctx_init_2}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx_1 = nn.Parameter(ctx_vectors_1.clone())
        self.ctx_2 = nn.Parameter(ctx_vectors_2.clone())

        self.meta_net = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
            ])
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ").lower() for name in classnames]

        enhanced_classnames = []
        for name in classnames:
            attr = PET_ATTRIBUTES.get(name, "")
            if attr:
             enhanced_classnames.append(f"{name}, {attr}")
            else:
                enhanced_classnames.append(name)

        name_lens = [len(_tokenizer.encode(name)) for name in enhanced_classnames]

        print("Using attribute-aware class prompts")
        for name, enhanced in zip(classnames[:5], enhanced_classnames[:5]):
            print(f"{name} -> {enhanced}")

        # tokenized prompts are based on template 1 for EOT positions / class token layout
        prompts = [ctx_init_1 + " " + name + "." for name in enhanced_classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])          # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # class tokens + EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )
        return prompts

    def forward(self, im_features, template_id=0):
        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx = self.ctx_1 if template_id == 0 else self.ctx_2

        bias = self.meta_net(im_features)   # (batch, ctx_dim)
        bias = bias.unsqueeze(1)            # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)              # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias            # (batch, n_ctx, ctx_dim)

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)      # (batch, n_cls, n_tkn, ctx_dim)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, template_id=0):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features, template_id=template_id)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)  # (batch, n_cls)
        return logits


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler("cuda") if cfg.TRAINER.COCOOP.PREC == "amp" else None

        self.lambda_cons = 30

        self.temp = 1 

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def _compute_losses(self, logits1, logits2, label):
        ce1 = F.cross_entropy(logits1, label)
        ce2 = F.cross_entropy(logits2, label)

        log_p2 = F.log_softmax(logits2 / self.temp, dim=1)
        p1 = F.softmax(logits1.detach() / self.temp, dim=1)
        cons_loss = F.kl_div(log_p2, p1, reduction="batchmean") * (self.temp ** 2)

        

        loss = ce1 + ce2 + self.lambda_cons * cons_loss
        return loss, ce1, ce2, cons_loss

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.COCOOP.PREC

        if prec == "amp":
            with autocast("cuda"):
                logits1 = model(image, template_id=0)
                logits2 = model(image, template_id=1)
                loss, ce1, ce2, cons_loss = self._compute_losses(logits1, logits2, label)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits1 = model(image, template_id=0)
            logits2 = model(image, template_id=1)
            loss, ce1, ce2, cons_loss = self._compute_losses(logits1, logits2, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item(),
            "ce1": ce1.item(),
            "ce2": ce2.item(),
            "cons": cons_loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        logits1 = self.model(input, template_id=0)
        logits2 = self.model(input, template_id=1)


        logits = (logits1 + logits2) / 2.0

        temperature = 1.2   
        logits = logits / temperature

        probs = F.softmax(logits, dim=1)

        return probs

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
