# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections

def get_Liver_wikg_config():
    """Returns the IRENE configuration."""
    config = ml_collections.ConfigDict()
    config.train_data_file = "../../docs/all_20240514.xlsx" # all_20240514 first_20240508
    config.test_data_file = "../../docs/second_20240508.xlsx"
    config.batch_size = 1
    config.dim_in = 384
    config.dim_hidden = 512
    config.topk = 6
    config.n_classes = 2
    config.agg_type='bi-interaction'
    config.dropout=0.3
    config.pool='attn'
    config.mode = "train"
    config.task = "二分类"
    config.label_dict = {"二分类": {"Non-desmoplastic": 0, "desmoplastic": 1}, "四分类":{"Predominant desmoplatic": 0, "Predominant replacement": 1, "Predominant pushing": 2, "Predominant mixed": 3, "Predominant replacement 1": 1, "Predominant replacement 2": 1}}
    config.eps = 100
    config.lr = 5e-5
    config.evaluation_save_path = "../../data/saved_models/test/evaluation.csv"
    config.model_save_path = "../../data/saved_models/test/"
    config.model_train_save_path = "../../data/saved_models/train/"
    config.results_path = '../../data/saved_resutlts'
    return config