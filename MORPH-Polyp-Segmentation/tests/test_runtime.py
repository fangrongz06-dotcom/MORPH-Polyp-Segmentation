import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from utils.common import load_config
from utils.dataset import PolypDataset, check_train_test_overlap
from scripts.package_release import release_files
from scripts.analyze_results import main as analyze_main

ROOT=Path(__file__).resolve().parents[1]


class SmallModel(torch.nn.Module):
    def __init__(self,config):
        super().__init__(); self.head=torch.nn.Conv2d(3,1,1)
    def forward(self,image,dense_click_map=None,valid_mask=None):
        return {"logits":self.head(image)}


def dataset_fixture(tmp_path):
    for folder in ("TrainDataset/image","TrainDataset/masks","TestDataset/Kvasir/images","TestDataset/Kvasir/masks"):
        (tmp_path/folder).mkdir(parents=True)
    mask=np.zeros((12,20),np.uint8); mask[3:9,5:15]=255
    for i,name in enumerate(("fixture_a.png","fixture_b.png")):
        Image.fromarray(np.full((12,20,3),40+i*30,np.uint8)).save(tmp_path/"TrainDataset/image"/name)
        Image.fromarray(mask).save(tmp_path/"TrainDataset/masks"/name)
    Image.fromarray(np.full((12,20,3),170,np.uint8)).save(tmp_path/"TestDataset/Kvasir/images/fixture_c.png")
    Image.fromarray(mask).save(tmp_path/"TestDataset/Kvasir/masks/fixture_c.png")
    train=tmp_path/"train.csv"; test=tmp_path/"test.csv"
    train.write_text("dataset,filename\nkvasir,fixture_a.png\nkvasir,fixture_b.png\n")
    test.write_text("dataset,filename\nkvasir,fixture_c.png\n")
    return train,test


def test_entrypoints_train_checkpoint_resume_and_all_protocols(tmp_path,monkeypatch):
    import models
    monkeypatch.setattr(models,"MORPH",SmallModel)
    train,test=dataset_fixture(tmp_path)
    config=load_config(ROOT/"configs/morph_1m.yaml")
    config["data"].update(geometry=64,train_manifest=str(train),test_manifests={"kvasir":str(test)})
    config["train"].update(epochs=2,learning_rate=.001)
    config["execution"].update(device="cpu",num_workers=0,pin_memory=False)
    config["output_dir"]=str(tmp_path/"runs")
    path=tmp_path/"config.yaml"; path.write_text(yaml.safe_dump(config))
    argv=["train.py","--config",str(path),"--data_root",str(tmp_path)]
    monkeypatch.setattr(sys,"argv",argv)
    importlib.import_module("train").main()
    checkpoint=tmp_path/"runs/final/1234/checkpoint_last.pth"
    saved=torch.load(checkpoint,weights_only=False)
    assert saved["epoch"]==2 and saved["config"]["train"]["phase"]=="final"
    monkeypatch.setattr(sys,"argv",argv+["--resume",str(checkpoint)])
    importlib.import_module("train").main()
    clicks={}
    for protocol in ("0-m","1-m","2-m"):
        output=tmp_path/protocol
        monkeypatch.setattr(sys,"argv",["test.py","--config",str(path),"--data_root",str(tmp_path),"--checkpoint",str(checkpoint),"--protocol",protocol,"--output",str(output)])
        importlib.import_module("test").main()
        summary=json.loads((output/"summary.json").read_text())
        assert summary["image_count"]==1 and summary["protocol"]==protocol
        clicks[protocol]=json.loads((output/"clicks.json").read_text())[0]["clicks"]
        assert len(clicks[protocol])<=int(protocol[0])
        assert (output/"evaluation_metadata.json").is_file()
    assert not clicks["0-m"] and clicks["1-m"][0]==clicks["2-m"][0]


def test_content_leak_guard_uses_pixels_not_only_filenames(tmp_path):
    train,test=dataset_fixture(tmp_path)
    Image.open(tmp_path/"TrainDataset/image/fixture_a.png").save(tmp_path/"TestDataset/Kvasir/images/fixture_c.png")
    with pytest.raises(ValueError,match="decoded-RGB"):
        check_train_test_overlap(PolypDataset(train,tmp_path,training=True),[PolypDataset(test,tmp_path)])


def test_odd_development_pool_and_heldout_evaluation(tmp_path,monkeypatch):
    import models
    monkeypatch.setattr(models,"MORPH",SmallModel)
    source,_=dataset_fixture(tmp_path)
    for filename,value in (("fixture_fit.png",110),("fixture_holdout.png",210)):
        Image.fromarray(np.full((12,20,3),value,np.uint8)).save(tmp_path/"TrainDataset/image"/filename)
        Image.fromarray(np.ones((12,20),np.uint8)*255).save(tmp_path/"TrainDataset/masks"/filename)
    fit=tmp_path/"fit.csv"; heldout=tmp_path/"heldout.csv"
    fit.write_text(source.read_text()+"kvasir,fixture_fit.png\n")
    heldout.write_text("dataset,filename\nkvasir,fixture_holdout.png\n")
    source.write_text(fit.read_text()+"kvasir,fixture_holdout.png\n")
    config=load_config(ROOT/"configs/morph_1m.yaml")
    config["data"].update(geometry=64,train_manifest=str(source),development_fit_manifest=str(fit),development_manifest=str(heldout))
    config["train"].update(epochs=1,learning_rate=.001)
    config["execution"].update(device="cpu",num_workers=0,pin_memory=False)
    config["output_dir"]=str(tmp_path/"runs")
    path=tmp_path/"config.yaml"; path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys,"argv",["train.py","--config",str(path),"--data_root",str(tmp_path),"--phase","development"])
    importlib.import_module("train").main()
    checkpoint=tmp_path/"runs/development/1234/checkpoint_last.pth"
    assert checkpoint.is_file()
    argv=["test.py","--config",str(path),"--data_root",str(tmp_path),"--checkpoint",str(checkpoint),"--protocol","0-m","--output",str(tmp_path/"dev_output")]
    monkeypatch.setattr(sys,"argv",argv)
    with pytest.raises(ValueError,match="Checkpoint phase"):
        importlib.import_module("test").main()
    monkeypatch.setattr(sys,"argv",argv+["--scope","development"])
    importlib.import_module("test").main()
    summary=json.loads((tmp_path/"dev_output/summary.json").read_text())
    assert summary["scope"]=="development" and summary["image_count"]==1


def test_release_contains_code_not_measurements_or_weights():
    files=release_files()
    assert files and not any(p.suffix in {".csv",".jsonl",".xlsx",".pth",".pdf"} for p in files)
    assert not any(".git" in p.relative_to(ROOT).parts for p in files)


def test_privacy_scan_allows_public_urls_and_blocks_drive_paths(tmp_path):
    document=tmp_path/"README.md"
    document.write_text("https://download.pytorch.org/whl/cpu")
    assert release_files(tmp_path)==[document]
    document.write_text(chr(67)+":"+chr(92)+"private"+chr(92)+"dataset")
    with pytest.raises(ValueError,match="Private path"):
        release_files(tmp_path)


def test_statistics_entrypoint_enforces_family_and_three_seed_coverage(tmp_path,monkeypatch):
    counts={"kvasir":100,"clinicdb":62,"colondb":380,"etis":196,"endoscene":60}
    audit=[]
    for dataset,count in counts.items():
        for i in range(count):
            audit.append({"dataset":dataset,"image_id":f"fixture_{i}","split":"evaluation",
                          "rgb_sha256":f"{dataset}_{i}","mask_sha256":"fixture_GT","content_eligible":True})
    audit_path=tmp_path/"audit.jsonl"
    audit_path.write_text("".join(json.dumps(r)+"\n" for r in audit))
    method_paths=[]
    for method in ("left","right"):
        files=[]
        for seed in (1234,2345,3456):
            folder=tmp_path/method/str(seed); folder.mkdir(parents=True)
            file=folder/"per_image_metrics.csv"
            with file.open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=["dataset","image_id","seed","protocol","dice","iou"])
                writer.writeheader()
                writer.writerows({"dataset":r["dataset"],"image_id":r["image_id"],"seed":seed,"protocol":"0-m","dice":.5,"iou":1/3} for r in audit)
            metadata={"seed":seed,"protocol":"0-m","scope":"benchmark","training_config":{},
                      "metrics_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"evaluation_config":{},
                      "geometry":1024,"preprocessing":"fixture","metric_space":"padded_grid","click_radius":5,"manifest_hashes":{}}
            (folder/"evaluation_metadata.json").write_text(json.dumps(metadata))
            files.append(str(file))
        method_paths.append(files)
    output=tmp_path/"statistics.json"
    argv=["analyze_results.py","--left",*method_paths[0],"--right",*method_paths[1],"--audit",str(audit_path),"--family","main_0m","--output",str(output)]
    monkeypatch.setattr(sys,"argv",argv)
    analyze_main()
    report=json.loads(output.read_text())
    assert report["holm_comparison_count"]==2
    assert all(r["N_pair"]==798 and r["N_zero"]==798 and r["n"]==0 and r["p_holm"]==1 for r in report["paired"])
