from scripts.review_asset_paths import asset_path

def test_manifest_windows_paths_are_relocated_without_rewriting_input(tmp_path):
    value=r'E:\3dconformergenerationcode\dataset\example\record.pt'
    assert asset_path(value,tmp_path)==tmp_path/'dataset/example/record.pt'
    assert asset_path(value.replace('\\','/'),tmp_path)==tmp_path/'dataset/example/record.pt'
    assert value==r'E:\3dconformergenerationcode\dataset\example\record.pt'

def test_unrelated_paths_are_preserved(tmp_path):
    value=tmp_path/'elsewhere.pt'
    assert asset_path(value,tmp_path)==value
