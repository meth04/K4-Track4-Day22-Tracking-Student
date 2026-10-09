"""Kiểm tra cấu hình chấm khi gói dữ liệu thiếu file JSON."""

import json

from evaluate_practice import _load_eval_config


def test_missing_config_uses_lab_fallback(tmp_path):
    assert _load_eval_config(tmp_path) == {'benchmark': 'LAB21', 'split': 'train'}


def test_pack_config_has_priority(tmp_path):
    directory = tmp_path / 'video_1'
    directory.mkdir()
    config = {'benchmark': 'CUSTOM', 'split': 'train'}
    (directory / 'eval_config.json').write_text(json.dumps(config), encoding='utf-8')
    assert _load_eval_config(tmp_path) == config
