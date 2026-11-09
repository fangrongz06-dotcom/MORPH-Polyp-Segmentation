import itertools
import random
import numpy as np
import pytest
import pywt
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from models.morph import MORPH, FrozenSAMViTBEncoder, FrozenResNet18Encoder, DensePromptProjection, ProgressiveDecoder
from models.whfc import WHFC
from utils.dataset import resize_and_pad, JointTransform, read_binary_mask, check_train_test_overlap
from utils.prompts import Click, sample_first_click, sample_second_click, render_dense_click_map
from utils.statistics import grouped_differences, signed_rank, repeated_run_ci, holm
from utils.metrics import compute_metrics


def test_aspect_ratio_and_exact_normalized_padding():
    image = Image.fromarray(np.full((8,16,3),100,np.uint8))
    mask = Image.fromarray(np.pad(np.full((4,8),255,np.uint8),((2,2),(4,4))))
    rgb,gt,valid,shape = resize_and_pad(image,mask,64)
    assert shape==(32,64) and gt.shape==(1,64,64)
    assert valid.sum()==32*64 and gt[:,32:].sum()==0
    assert torch.equal(rgb[:,40,40],torch.tensor([.485,.456,.406]))
    sam = FrozenSAMViTBEncoder.preprocess(rgb[None],valid[None])
    assert torch.count_nonzero(sam[:,:,32:])==0
    cnn = FrozenResNet18Encoder(pretrained=False).eval()
    captured = []
    hook = cnn.stem.register_forward_pre_hook(lambda module,args:captured.append(args[0].clone()))
    cnn(rgb[None],valid[None])
    hook.remove()
    assert torch.count_nonzero(captured[0][:,:,32:])==0


def test_brightness_and_contrast_draws_are_independent(monkeypatch):
    from utils.dataset import TF
    draws = iter([.9,.9,.1,.9])
    monkeypatch.setattr(random,"random",lambda:next(draws))
    called=[]
    monkeypatch.setattr(TF,"adjust_brightness",lambda image,factor:(called.append("brightness") or image))
    monkeypatch.setattr(TF,"adjust_contrast",lambda image,factor:(called.append("contrast") or image))
    image=Image.fromarray(np.full((8,8,3),100,np.uint8))
    mask=Image.fromarray(np.zeros((8,8),np.uint8))
    JointTransform(0,0,[-10,10],.1,rotation_probability=0,intensity_probability=.5)(image,mask)
    assert called==["brightness"]


def test_whfc_against_independent_pywavelets():
    rng=np.random.default_rng(5)
    image=rng.random((3,64,64),dtype=np.float32)
    model=WHFC()
    features,aux=model(torch.from_numpy(image)[None])
    expected=[[] for _ in range(3)]
    for channel in image:
        current=channel
        for level in range(3):
            current,bands=pywt.dwt2(current,"haar",mode="symmetric")
            expected[level].append(bands)
    for level,actual in enumerate(aux["details"]):
        for direction in range(3):
            reference=np.stack([c[direction] for c in expected[level]])
            np.testing.assert_allclose(actual[direction][0].detach().numpy(),reference,atol=2e-6)
    assert [f.shape[1] for f in features]==[64,128,256,512]
    assert aux["basis"].shape==(1,27,64,64)
    assert not list(model.dwt.parameters())
    assert all(p.requires_grad for projection in model.projections for p in projection.parameters())
    assert any(torch.any(band<0) for bands in aux["details"] for band in bands)


def test_click_polarity_ties_and_error_free_case():
    gt=np.zeros((16,16),bool); gt[2:4,2:4]=True
    prediction=np.zeros_like(gt); prediction[10:12,10:12]=True
    click=sample_second_click(gt,prediction,np.random.default_rng(3))
    assert click.positive and gt[click.y,click.x]
    assert sample_second_click(gt,gt,np.random.default_rng(3)) is None
    first=sample_first_click(gt,np.random.default_rng(1234))
    dense=render_dense_click_map([[first,Click(12,12,False)]],image_size=16)
    assert dense.shape==(1,2,128,128) and dense[:,0].sum()>0 and dense[:,1].sum()>0


def test_zero_prompt_is_exact_identity_after_batchnorm_training():
    projection=DensePromptProjection().train()
    projection(torch.rand(2,2,128,128))
    assert torch.count_nonzero(projection(torch.zeros(2,2,128,128)))==0


class FeatureCNN(nn.Module):
    def forward(self,image,valid_mask=None):
        gray=image.mean(1,keepdim=True)
        return tuple(F.avg_pool2d(gray,scale).expand(-1,c,-1,-1).detach() for c,scale in zip((64,128,256,512),(4,8,16,32)))


class FeatureViT(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.Identity()
    def forward(self,image,prompt_fn,valid_mask=None):
        x=F.adaptive_avg_pool2d(image.mean(1,keepdim=True),(64,64)).expand(-1,768,-1,-1)
        prompts={}
        for block in (2,5,8,11):
            prompt=prompt_fn(block,x)
            prompts[block]=prompt
            x=(x+prompt)*1.01
        return x,prompts


def test_trainable_trunk_gradients_through_four_prompt_injections():
    model=MORPH({"model":{"sam_checkpoint":"unused"}},cnn=FeatureCNN(),vit=FeatureViT())
    # Keep this unit test small; full native decoder inference is checked separately.
    model.decoder=nn.Conv2d(512,1,1)
    image=torch.rand(2,3,1024,1024)
    dense=render_dense_click_map([[Click(500,500,True)],[Click(510,510,True)]])
    result=model(image,dense)
    result["logits"].square().mean().backward()
    assert set(result["prompts"])=={2,5,8,11}
    for module in (model.whfc.projections,*model.meaas,model.cbaf,model.dense_prompt,model.decoder):
        gradients=[p.grad for p in module.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)


def test_all_five_decoder_stages_backpropagate():
    decoder=ProgressiveDecoder().train()
    x=torch.randn(2,512,1,1,requires_grad=True)
    output=decoder(x)
    assert output.shape==(2,1,32,32)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_raw_bkai_color_jpeg_cannot_be_silently_binarized(tmp_path):
    path=tmp_path/"label.jpeg"
    rgb=np.zeros((8,8,3),np.uint8); rgb[2:6,2:6,0]=255
    Image.fromarray(rgb).save(path)
    with Image.open(path) as image,pytest.raises(ValueError,match="binary GT"):
        read_binary_mask(image,"bkai")


def test_zero_differences_and_exact_signed_rank_tails():
    result=signed_rank([0.,.1,.2,.3])
    assert result["N_pair"]==4 and result["N_zero"]==1 and result["n"]==3
    assert result["p_value"]==.25 and result["rank_biserial"]==1
    ties=signed_rank([1.,1.,-1.,0.])
    assert ties["W_plus"]==4 and ties["W_minus"]==2 and ties["p_value"]==1
    assert signed_rank([0.,0.])["p_value"]==1
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])


def test_binary_metrics_follow_set_definitions_without_score_smoothing():
    target=np.array([[1,1],[0,0]],bool)
    prediction=np.array([[1,0],[1,0]],bool)
    metrics=compute_metrics(prediction,target)
    assert metrics["dice"]==.5 and metrics["iou"]==1/3
    assert compute_metrics(np.zeros_like(target),target)["dice"]==0
    assert compute_metrics(np.zeros_like(target),np.zeros_like(target))["dice"]==1


def test_three_run_mean_precedes_content_grouping_and_never_inflates_n():
    audit=[{"dataset":"sample","image_id":str(i),"split":"evaluation","rgb_sha256":"shared" if i<2 else "other", "mask_sha256":"same", "content_eligible":True} for i in range(3)]
    left,right=[],[]
    for seed in (1234,2345,3456):
        for i in range(3):
            row={"dataset":"sample","image_id":str(i),"seed":seed,"protocol":"0-m"}
            left.append({**row,"dice":float(i!=1)})
            right.append({**row,"dice":0.})
    grouped,counts=grouped_differences(left,right,audit,"0-m","dice")
    assert len(grouped)==2 and sorted(r["difference"] for r in grouped)==[.5,1.]
    assert counts["benchmark_records_per_run"]==3
    with pytest.raises(ValueError,match="exactly all"):
        grouped_differences(left[:-1],right,audit,"0-m","dice")
    ci=repeated_run_ci([1,2,3])
    assert ci["mean"]==2 and ci["sample_sd"]==1
    assert ci["ci95"][1]==pytest.approx(2+4.3026527299/np.sqrt(3))
